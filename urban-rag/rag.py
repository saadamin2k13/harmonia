import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime

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
        chunk_overlap: int = 50,
        embedding_model: str = "all-mpnet-base-v2",
        llm_model: str = "google/gemma-2b-it",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Urban Analysis RAG system.
        """
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
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
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving meaningful boundaries.
        """
        sections = re.split(r'(?:\n\s*\n|\={3,})', text)
        return [s.strip() for s in sections if s.strip()]

    def process_text_data(self, text: str) -> None:
        """
        Process raw text data and organize it into analyzed sections.
        """
        self.text_chunks = self._chunk_text(text)
        
        # Create embeddings for all chunks
        self.embeddings = self.embedding_model.encode(
            self.text_chunks,
            convert_to_tensor=True,
            device=self.device
        )

    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant text chunks based on query.
        """
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        
        similarity_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings)
        top_k_indices = torch.topk(similarity_scores, min(top_k, len(self.text_chunks))).indices
        
        return [self.text_chunks[int(i)] for i in top_k_indices]

    def generate_response(self, query: str) -> str:
        """
        Generate a response using retrieved context and an LLM.
        """
        relevant_chunks = self.retrieve_relevant_context(query, top_k=5)
        
        if not relevant_chunks:
            return "No relevant information found in the dataset."
        
        context = "\n\n".join(relevant_chunks)
        prompt = f"""Based on the following urban analysis data:

{context}

Question: {query}

Please provide a detailed response strictly based on the provided context."""
        
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        outputs = self.llm.generate(
            **input_ids,
            temperature=0.7,
            max_new_tokens=512,
            do_sample=True
        )
        
        return f"Retrieved Context:\n\n{context}\n\nResponse:\n" + self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Example usage
def main():
    # Initialize the RAG system
    rag = UrbanAnalysisRAG()
    
    # Load and process text data
    with open("verbalized-data.txt", "r") as f:
        text_data = f.read()
    
    rag.process_text_data(text_data)
    
    # Example queries
    queries = [
        "How the length of lines per capita relates to the number of lines in the area?",
        "Analyze the temporal changes in Turin district, particularly focusing on how immigrant percentage has changed?",
        "How the number of stops per capita and number of lines stopping reflect the population size in Turin district?", 
        #"Whether district's transport coverage appears appropriate for its demographic compositions?",
        "How does public transport accessibility differ between areas with varying age distributions?",
        "Which districts have the lowest number of stops per capita, and how does this impact accessibility?",
        "Are there areas where public transport coverage is disproportionately low compared to demand?",
        
        #"How does the coverage of stoppings and number of lines stopping reflect the number of people in a census?",
        #"Does a higher number of minors and seniors in a census reflect a higher number of stops and lines stopping?",
        #"Does the length of the lines stopping per population increase when the number of lines is higher?"
    ]
    
    # Generate responses
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        response = rag.generate_response(query)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()

