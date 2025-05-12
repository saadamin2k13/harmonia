import os
import torch
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt_tab')

from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class ChunkMetadata:
    """Metadata for each text chunk"""
    chunk_id: str
    source_section: str
    location: Optional[str] = None
    timestamp: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)

@dataclass
class AnalysisMetadata:
    """Metadata for each analysis section"""
    section_id: str
    analysis_type: str
    timestamp: datetime
    location: str
    metrics: Dict[str, float]
    raw_text: str

class UrbanAnalysisRAG:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "all-mpnet-base-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model: str = "google/gemma-2b-it",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize the Urban Analysis RAG system with advanced features.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks to maintain context
            embedding_model: SentenceTransformer model for embeddings
            cross_encoder_model: Model for reranking
            llm_model: Language model for generation
            device: Computing device (cuda/cpu)
            max_new_tokens: Maximum new tokens to generate
            temperature: Temperature for generation
        """
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Initializing UrbanAnalysisRAG with device: {device}")
        
        # Initialize models
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=self.device)
        
        logger.info(f"Loading LLM: {llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Storage for processed data
        self.analysis_sections: Dict[str, AnalysisMetadata] = {}
        self.embeddings = None
        self.text_chunks = []
        self.chunk_metadata: List[ChunkMetadata] = []
        
        # Keyword extraction
        self.keyword_extractor = pipeline(
            "token-classification", 
            model="dslim/bert-base-NER", 
            aggregation_strategy="simple"
        )
        
        # Performance metrics
        self.metrics = {
            "retrieval_time": [],
            "generation_time": [],
            "total_response_time": [],
        }
        
        # Cache for queries
        self.query_cache = {}

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text chunk using pattern matching and NER.
        """
        metadata = {}
        
        # Look for location information
        location_match = re.search(r'(?i)(?:in|at|for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:district|area|region|zone)', text)
        if location_match:
            metadata["location"] = location_match.group(1)
            
        # Look for dates
        date_match = re.search(r'(?i)(?:in|from|during|since)\s+(\d{4}(?:-\d{2}-\d{2})?)', text)
        if date_match:
            try:
                metadata["date"] = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            except ValueError:
                try:
                    metadata["date"] = datetime.strptime(date_match.group(1), "%Y")
                except ValueError:
                    pass
        
        # Extract metrics with numbers
        metric_matches = re.findall(r'([a-zA-Z\s]+(?:rate|percentage|ratio|index|count|number|density))(?:\s+is|\s*:)?\s+(\d+\.?\d*)\s*%?', text)
        for metric, value in metric_matches:
            metadata[metric.strip().lower().replace(" ", "_")] = float(value)
            
        # Extract named entities
        try:
            entities = self.keyword_extractor(text)
            keywords = [item["word"] for item in entities if item["entity_group"] in ["LOC", "ORG", "MISC"]]
            if keywords:
                metadata["keywords"] = keywords
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            
        return metadata

    def _chunk_text_smart(self, text: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """
        Split text into chunks while preserving semantic boundaries and extracting metadata.
        """
        # First split by major sections
        sections = re.split(r'(?:\n\s*\n|\={3,}|\#{3,})', text)
        sections = [s.strip() for s in sections if s.strip()]
        
        chunks = []
        chunk_metadata_list = []
        
        for section_idx, section in enumerate(sections):
            # Get section title if available
            section_title = f"Section {section_idx+1}"
            title_match = re.match(r'^(.*?)(?:\n|$)', section)
            if title_match:
                potential_title = title_match.group(1).strip()
                if len(potential_title) < 100 and not re.match(r'^\d+\.\s+', potential_title):
                    section_title = potential_title
            
            # Split section into sentences
            sentences = sent_tokenize(section)
            
            # Group sentences into chunks
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_len = len(sentence)
                
                # If adding this sentence exceeds chunk size and we already have content,
                # finalize the current chunk and start a new one
                if current_length + sentence_len > self.chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Extract metadata from the chunk
                    metadata = self._extract_metadata(chunk_text)
                    chunk_metadata = ChunkMetadata(
                        chunk_id=f"chunk_{len(chunks)}",
                        source_section=section_title,
                        location=metadata.get("location"),
                        timestamp=metadata.get("date"),
                        metrics={k: v for k, v in metadata.items() 
                                if k not in ["location", "date", "keywords"]},
                        keywords=metadata.get("keywords", [])
                    )
                    chunk_metadata_list.append(chunk_metadata)
                    
                    # Start new chunk, possibly including overlap
                    overlap_size = min(self.chunk_overlap, len(current_chunk))
                    current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                    current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                
                current_chunk.append(sentence)
                current_length += sentence_len + 1
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Extract metadata for the last chunk
                metadata = self._extract_metadata(chunk_text)
                chunk_metadata = ChunkMetadata(
                    chunk_id=f"chunk_{len(chunks)}",
                    source_section=section_title,
                    location=metadata.get("location"),
                    timestamp=metadata.get("date"),
                    metrics={k: v for k, v in metadata.items() 
                            if k not in ["location", "date", "keywords"]},
                    keywords=metadata.get("keywords", [])
                )
                chunk_metadata_list.append(chunk_metadata)
        
        return chunks, chunk_metadata_list

    def process_text_data(self, text: str) -> None:
        """
        Process raw text data with smart chunking and metadata extraction.
        """
        logger.info("Processing text data and creating embeddings")
        
        # Smart chunking with metadata extraction
        self.text_chunks, self.chunk_metadata = self._chunk_text_smart(text)
        logger.info(f"Created {len(self.text_chunks)} chunks with metadata")
        
        # Create embeddings for all chunks
        self.embeddings = self.embedding_model.encode(
            self.text_chunks,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=True
        )
        logger.info("Created embeddings for all chunks")
        
        # Create a summary of available information
        self._create_data_summary()

    def _create_data_summary(self) -> None:
        """
        Create a summary of the available data for better query routing.
        """
        # Extract unique locations
        locations = set()
        metrics = set()
        for metadata in self.chunk_metadata:
            if metadata.location:
                locations.add(metadata.location)
            metrics.update(metadata.metrics.keys())
            
        self.data_summary = {
            "locations": sorted(list(locations)),
            "available_metrics": sorted(list(metrics)),
            "total_chunks": len(self.text_chunks)
        }
        
        logger.info(f"Data summary created. Locations: {len(locations)}, Metrics: {len(metrics)}")

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query to improve retrieval by generating variations.
        """
        expanded_queries = [query]
        
        # Add a variation without question marks
        if '?' in query:
            expanded_queries.append(query.replace('?', ''))
            
        # Add variations with key terms from the query
        important_terms = re.findall(r'\b(?:transport|line|stop|population|district|area|coverage|accessibility|demographic|senior|minor|immigrant|percentage|ratio|capita)\w*\b', query.lower())
        if important_terms:
            for term in important_terms[:3]:  # Limit to top 3 terms
                expanded_queries.append(f"{term} {query}")
                
        return expanded_queries

    def retrieve_relevant_context(self, query: str, top_k: int = 5, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant text chunks based on query using hybrid retrieval and reranking.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            use_reranking: Whether to use cross-encoder for reranking
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
        import time
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}_{top_k}_{use_reranking}"
        if cache_key in self.query_cache:
            logger.info("Using cached retrieval results")
            return self.query_cache[cache_key]
        
        # Expand query for better retrieval
        expanded_queries = self._expand_query(query)
        logger.info(f"Expanded query into {len(expanded_queries)} variations")
        
        all_candidate_indices = set()
        all_scores = {}
        
        # Process each expanded query
        for expanded_query in expanded_queries:
            query_embedding = self.embedding_model.encode(
                expanded_query,
                convert_to_tensor=True,
                device=self.device
            )
            
            # Semantic search
            similarity_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings)
            
            # Get top candidates
            candidate_count = min(top_k * 2, len(self.text_chunks))  # Get more candidates for reranking
            top_indices = torch.topk(similarity_scores, candidate_count).indices
            
            # Add to our candidate pool
            for idx in top_indices:
                idx_int = int(idx)
                all_candidate_indices.add(idx_int)
                all_scores[idx_int] = max(all_scores.get(idx_int, 0), float(similarity_scores[idx_int]))
        
        # Perform lexical search to complement semantic search
        # Simple keyword matching for demonstration
        keywords = re.findall(r'\b\w{4,}\b', query.lower())
        if keywords:
            for idx, chunk in enumerate(self.text_chunks):
                keyword_matches = sum(1 for keyword in keywords if keyword in chunk.lower())
                if keyword_matches > 0 and idx not in all_candidate_indices and len(all_candidate_indices) < top_k * 3:
                    all_candidate_indices.add(idx)
                    all_scores[idx] = keyword_matches / len(keywords) * 0.8  # Slightly lower weight than semantic
        
        candidate_indices = list(all_candidate_indices)
        
        # Reranking step
        if use_reranking and len(candidate_indices) > top_k:
            logger.info(f"Reranking {len(candidate_indices)} candidates")
            
            # Prepare cross-encoder inputs
            cross_inp = [(query, self.text_chunks[idx]) for idx in candidate_indices]
            cross_scores = self.cross_encoder.predict(cross_inp)
            
            # Sort by cross-encoder scores
            reranked_indices = [candidate_indices[i] for i in np.argsort(-cross_scores)[:top_k]]
            selected_indices = reranked_indices
        else:
            # Sort by original scores
            selected_indices = sorted(candidate_indices, key=lambda idx: all_scores.get(idx, 0), reverse=True)[:top_k]
        
        # Create result with chunks and metadata
        result = []
        for idx in selected_indices:
            result.append({
                "text": self.text_chunks[idx],
                "score": all_scores.get(idx, 0) if not use_reranking else float(cross_scores[candidate_indices.index(idx)]),
                "metadata": self.chunk_metadata[idx]
            })
        
        # Add to cache
        self.query_cache[cache_key] = result
        
        # Log retrieval time
        retrieval_time = time.time() - start_time
        self.metrics["retrieval_time"].append(retrieval_time)
        logger.info(f"Retrieved {len(result)} chunks in {retrieval_time:.2f} seconds")
        
        return result

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context for the LLM.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Extract location if available
            location_str = f" from {chunk['metadata'].location}" if chunk['metadata'].location else ""
            
            # Format chunk with metadata and citation
            chunk_text = f"[Chunk {i+1}{location_str}]:\n{chunk['text']}\n"
            context_parts.append(chunk_text)
            
        return "\n".join(context_parts)

    def _generate_evidence_context(self, chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Create an evidence-focused context with fact extraction.
        """
        # Group chunks by location
        locations = {}
        for chunk in chunks:
            location = chunk['metadata'].location or "Unknown"
            if location not in locations:
                locations[location] = []
            locations[location].append(chunk)
        
        # Extract facts and metrics from chunks
        facts = []
        metrics_by_location = {}
        
        for location, loc_chunks in locations.items():
            metrics_by_location[location] = {}
            
            for chunk in loc_chunks:
                # Add metrics from metadata
                for metric_name, metric_value in chunk['metadata'].metrics.items():
                    if metric_name not in metrics_by_location[location]:
                        metrics_by_location[location][metric_name] = metric_value
                
                # Extract simple facts using patterns
                fact_patterns = [
                    r'([A-Z][^.!?]*(?:increased|decreased|remained|maintained|had)[^.!?]*(?:by|to|at|from)[^.!?]*\d+[^.!?]*[.!?])',
                    r'([A-Z][^.!?]*(?:higher|lower|greater|less|more|fewer)[^.!?]*(?:than|compared to|versus)[^.!?]*[.!?])',
                    r'([A-Z][^.!?]*(?:rate|ratio|percentage|number)[^.!?]*(?:is|was|of)[^.!?]*\d+[^.!?]*[.!?])'
                ]
                
                for pattern in fact_patterns:
                    matches = re.findall(pattern, chunk['text'])
                    facts.extend(matches)
        
        # Deduplicate facts
        facts = list(set(facts))
        
        return {
            "grouped_by_location": locations,
            "facts": facts[:15],  # Limit number of facts
            "metrics": metrics_by_location
        }

    def generate_response(self, query: str, top_k: int = 5, use_reranking: bool = True) -> str:
        """
        Generate a response using retrieved context and an LLM with improved prompt engineering.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_reranking: Whether to use reranking
            
        Returns:
            Generated response with citation information
        """
        import time
        start_time = time.time()
        
        relevant_chunks = self.retrieve_relevant_context(query, top_k=top_k, use_reranking=use_reranking)
        
        if not relevant_chunks:
            return "No relevant information found in the dataset."
        
        # Create standard context
        context = self._format_context(relevant_chunks)
        
        # Create evidence context with extracted facts and metrics
        evidence = self._generate_evidence_context(relevant_chunks, query)
        
        # Format facts as bulleted list
        facts_list = "\n".join([f"• {fact}" for fact in evidence["facts"]])
        
        # Format metrics by location
        metrics_section = ""
        for location, metrics in evidence["metrics"].items():
            if metrics:
                metrics_str = ", ".join([f"{name}: {value}" for name, value in metrics.items()])
                metrics_section += f"• {location}: {metrics_str}\n"
        
        # Advanced prompt with evidence, facts extraction, and instruction
        prompt = f"""You are an urban transportation analysis expert examining data about different districts of a city.

Below is the relevant information retrieved from the transportation database:

{context}

Key facts extracted from the data:
{facts_list}

Key metrics by location:
{metrics_section}

Here is a summary of the types of data available:
- Locations mentioned: {', '.join(self.data_summary.get('locations', ['Unknown']))}
- Metrics available: {', '.join(self.data_summary.get('available_metrics', ['Unknown']))}

Question: {query}

Instructions:
1. Answer based ONLY on the provided information
2. If the data is insufficient to answer completely, acknowledge this explicitly
3. Point out any apparent inconsistencies in the data
4. Include specific statistics and metrics from the data when relevant
5. Cite the specific chunks you're drawing information from using [Chunk X] notation
6. When comparing districts, use concrete numbers rather than vague comparisons
7. Begin with a concise summary, then provide detailed analysis
8. Conclude with key implications of this transportation data

Your response:"""

        logger.info("Generating response with LLM")
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        outputs = self.llm.generate(
            **input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        
        # Add confidence assessment
        confidence_level = "High" if len(evidence["facts"]) >= 5 and len(relevant_chunks) >= 3 else "Medium" if len(evidence["facts"]) >= 2 else "Low"
        
        # Format the final response
        final_response = f"""Response:

{response}

---
Sources: This response is based on {len(relevant_chunks)} relevant data chunks about urban transportation.
Confidence: {confidence_level} (based on {len(evidence['facts'])} supporting facts)
"""
        
        # Log generation time
        generation_time = time.time() - start_time
        self.metrics["generation_time"].append(generation_time)
        self.metrics["total_response_time"].append(generation_time)
        
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        
        return final_response

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get average performance metrics for the RAG system.
        """
        metrics = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                metrics[f"avg_{metric_name}"] = sum(values) / len(values)
                metrics[f"max_{metric_name}"] = max(values)
        
        return metrics

    def evaluate_response(self, query: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the quality of a response (for development purposes).
        """
        evaluation = {}
        
        # Analyze citation usage
        citation_count = len(re.findall(r'\[Chunk \d+\]', response))
        evaluation["citation_count"] = citation_count
        
        # Check for hedging language suggesting uncertainty
        hedging_phrases = ["may", "might", "could be", "possibly", "perhaps", "it seems", "likely", "unlikely"]
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in response.lower())
        evaluation["hedging_score"] = hedging_count / len(hedging_phrases)
        
        # Check for specific metrics mentioned
        metrics_mentioned = len(re.findall(r'\d+\.\d+%|\d+%|\d+\.\d+|\b\d+\b', response))
        evaluation["metrics_count"] = metrics_mentioned
        
        return evaluation

# Example usage
def main():
    # Initialize the RAG system
    rag = UrbanAnalysisRAG()

    # Load and process text data
    try:
        with open("verbalized-data.txt", "r") as f:
            text_data = f.read()
    except FileNotFoundError:
        logger.error("verbalized-data.txt not found. Please provide the file.")
        return

    logger.info("Processing text data...")
    rag.process_text_data(text_data)
    logger.info("Data processing complete")

    # Example queries
    queries = [
        "How the length of lines per capita relates to the number of lines in the area?",
        "Analyze the temporal changes in Turin district, particularly focusing on how immigrant percentage has changed?",
        "How the number of stops per capita and number of lines stopping reflect the population size in Turin district?",
        "How does public transport accessibility differ between areas with varying age distributions?",
        "Which districts have the lowest number of stops per capita, and how does this impact accessibility?",
        "Are there areas where public transport coverage is disproportionately low compared to demand?",
    ]

    # Generate responses
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        # Generate response with reranking
        response = rag.generate_response(query, top_k=5, use_reranking=True)
        print(response)
        
        # Evaluate response quality
        evaluation = rag.evaluate_response(query, response)
        print("\nResponse Evaluation:")
        for metric, value in evaluation.items():
            print(f"- {metric}: {value}")
        
        print("-" * 80)

    # Print performance metrics
    perf_metrics = rag.get_performance_metrics()
    print("\nSystem Performance Metrics:")
    for metric, value in perf_metrics.items():
        print(f"{metric}: {value:.4f} seconds")

if __name__ == "__main__":
    main()
