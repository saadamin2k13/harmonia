import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class TurinCensusRAG:
    def __init__(self, json_path):
        """Initialize the RAG system with a JSON file containing Turin census data."""
        # Load the JSON data
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Ensure data is in list format
            if not isinstance(data, list):
                self.data = [data]  # Convert single object to list
            else:
                self.data = data
                
            print(f"Successfully loaded {len(self.data)} records from {json_path}")
        except Exception as e:
            print(f"Error loading JSON data: {str(e)}")
            # Initialize with empty list if file loading fails
            self.data = []

        # Keep the original data in a dictionary for easy lookup
        self.original_data = {}
        for item in self.data:
            # Create a unique key for each record
            key = item["id"]
            self.original_data[key] = item
            
            # Also create secondary keys for more flexible lookups
            if "metadata" in item:
                metadata = item["metadata"]
                secondary_key = (metadata.get("census_id"), metadata.get("year"), metadata.get("district"))
                self.original_data[secondary_key] = item

        # Extract text and metadata for retrieval
        self.documents = []
        for item in self.data:
            # Add the main text field as a document
            if "text" in item:
                doc = {
                    "text": item["text"],
                    "id": item["id"]
                }
                
                # Add metadata if available
                if "metadata" in item:
                    doc["census_id"] = item["metadata"].get("census_id")
                    doc["year"] = item["metadata"].get("year")
                    doc["zone"] = item["metadata"].get("zone")
                    doc["district"] = item["metadata"].get("district")
                
                self.documents.append(doc)
                
                # If there's raw_data, add that formatted as text too for better retrieval
                if "raw_data" in item:
                    raw_text = self._create_raw_data_text(item["raw_data"])
                    if raw_text:
                        raw_doc = doc.copy()
                        raw_doc["text"] = raw_text
                        self.documents.append(raw_doc)

        if not self.documents:
            print("Warning: No documents were created for retrieval")
            return

        # Use TF-IDF vectorizer for document retrieval
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_texts = [doc["text"] for doc in self.documents]
        
        try:
            self.document_vectors = self.vectorizer.fit_transform(self.document_texts)
            print(f"Created TF-IDF vectors for {len(self.documents)} document chunks")
        except Exception as e:
            print(f"Error creating document vectors: {str(e)}")
            self.document_vectors = None

    def _create_raw_data_text(self, raw_data):
        """Convert raw_data fields into a descriptive text for better retrieval."""
        text_parts = []
        for key, value in raw_data.items():
            # Format the key to be more readable
            readable_key = key.replace("_", " ").replace("tot", "total")
            
            # Add to text parts
            text_parts.append(f"{readable_key}: {value}")
        
        return ". ".join(text_parts)

    def retrieve(self, query, top_k=5):
        """Retrieve the most relevant document chunks for a query."""
        if not self.documents or self.document_vectors is None:
            return []
            
        try:
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])

            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.document_vectors)[0]

            # Get top_k indices
            top_indices = np.argsort(similarity_scores)[-top_k:][::-1]

            # Return top_k documents with their scores
            top_docs = []
            for idx in top_indices:
                if similarity_scores[idx] > 0.05:  # Only include if score is significant
                    top_docs.append({
                        "document": self.documents[idx],
                        "score": similarity_scores[idx]
                    })
                    
            print(f"Retrieved {len(top_docs)} documents with scores: {[round(d['score'], 3) for d in top_docs]}")
            return top_docs
        except Exception as e:
            print(f"Error in retrieve: {str(e)}")
            return []

    def extract_key_info(self, retrieved_docs):
        """Extract key information from retrieved documents."""
        if not retrieved_docs:
            return None
            
        # Check if any document has a good enough score
        best_doc = retrieved_docs[0]
        if best_doc["score"] < 0.1:
            print(f"Best document score {best_doc['score']} is too low for reliable information")
            return None
            
        doc_id = best_doc["document"]["id"]
        print(f"Selected document with ID: {doc_id}")
        
        # Get the complete record using the id
        return self.original_data.get(doc_id)
        
    def _verify_location_match(self, query, zone):
        """Verify if the location in the query matches the retrieved document."""
        # Extract potential location names from query
        query_lower = query.lower()
        locations = ["municipio", "centro", "vanchiglia", "san salvario"]  # Add common Turin areas
        
        for location in locations:
            if location in query_lower:
                # If query mentions a location, check if it matches retrieved zone
                if location != zone.lower() and zone.lower() != "n/a":
                    print(f"Location mismatch: query mentions {location} but document is about {zone}")
                    return False
        
        return True

    def answer_question(self, query):
        """Generate a polished answer for the user's question."""
        query_lower = query.lower()
        
        # Check if question is about Turin or Italy
        if not any(term in query_lower for term in ["turin", "torino", "italy", "italian", "census", "district", "municipio"]):
            locations = ["wazirabad", "delhi", "london", "paris", "new york"]
            for location in locations:
                if location in query_lower:
                    return f"I don't have information about {location.title()} in my dataset. I can only answer questions about census areas in Turin, Italy."
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=3)
        
        if not retrieved_docs:
            return "I don't have enough information in my dataset to answer this question about Turin's census data. My dataset is limited to specific census areas and years in Turin, Italy."

        # Get the most relevant complete entry
        top_entry = self.extract_key_info(retrieved_docs)
        if not top_entry:
            return "While I found some related information, I couldn't find specific data that directly answers your question about Turin's census areas."

        # Print the entry data for debugging
        print(f"Using entry: {top_entry['id']}")
        
        # Extract key information from the entry
        metadata = top_entry.get("metadata", {})
        raw_data = top_entry.get("raw_data", {})
        
        # Basic information always included
        census_id = metadata.get("census_id", "N/A")
        year = metadata.get("year", "N/A")
        zone = metadata.get("zone", "N/A")
        district = metadata.get("district", "N/A")
        
        # Verify if the location in query matches our retrieved document
        if not self._verify_location_match(query_lower, zone):
            # If there's a location mismatch, give a more specific "no data" response
            for location in ["wazirabad", "delhi", "london", "paris", "new york"]:
                if location in query_lower:
                    return f"I don't have information about {location.title()}. My dataset only contains information about census areas in Turin, Italy."
            
            # If query mentions a Turin location not in our top document
            for location in ["municipio", "centro", "vanchiglia", "san salvario"]:
                if location in query_lower and location.lower() != zone.lower():
                    return f"I don't have specific information that answers your question about {location.title()}. The closest matching data I have is for {zone} (District {district}) in {year}."
        
        # Generate response based on question type
        if any(term in query_lower for term in ["population", "residents", "people", "demographic"]):
            response = self._answer_population_question(query_lower, census_id, year, zone, district, raw_data)
        
        elif any(term in query_lower for term in ["transport", "transit", "bus", "tram", "stop", "line"]):
            response = self._answer_transport_question(query_lower, census_id, year, zone, district, raw_data, metadata)
        
        elif any(term in query_lower for term in ["accident", "crash", "vehicle", "injury"]):
            response = self._answer_accident_question(query_lower, census_id, year, zone, district, raw_data, metadata)
        
        elif any(term in query_lower for term in ["housing", "home", "house", "family", "families"]):
            response = self._answer_housing_question(query_lower, census_id, year, zone, district, raw_data)
        
        else:
            response = self._answer_general_question(query_lower, census_id, year, zone, district, raw_data, metadata)
        
        return response
    
    def _answer_population_question(self, query, census_id, year, zone, district, raw_data):
        """Generate response for population-related questions."""
        total_pop = raw_data.get("tot", "N/A")
        female_pop = raw_data.get("tot_F", "N/A")
        male_pop = raw_data.get("tot_M", "N/A")
        foreign_pop = raw_data.get("tot_foreig", "N/A")
        minors = raw_data.get("minors", "N/A")
        seniors = raw_data.get("seniors", "N/A")
        working_age = raw_data.get("working_age", "N/A")
        
        response = f"In {year}, census area {census_id} ({zone}, District {district}) had a total population of {total_pop} residents"
        
        # Add gender breakdown if it seems relevant
        if "gender" in query or "male" in query or "female" in query or "sex" in query:
            response += f", with {female_pop} females and {male_pop} males"
        
        response += "."
        
        # Add age demographics if relevant
        if "age" in query or "demographic" in query or "minor" in query or "senior" in query or "child" in query or "elder" in query:
            response += f" The age distribution showed {minors} minors, {working_age} working-age adults, and {seniors} seniors."
        
        # Add foreign population if relevant
        if "foreign" in query or "immigrant" in query or "international" in query:
            if foreign_pop != "N/A" and total_pop != "N/A" and float(total_pop) > 0:
                foreign_percentage = round((float(foreign_pop) / float(total_pop)) * 100, 1)
                response += f" There were {foreign_pop} foreign residents, representing approximately {foreign_percentage}% of the total population."
            else:
                response += f" There were {foreign_pop} foreign residents in this area."
        
        return response
    
    def _answer_transport_question(self, query, census_id, year, zone, district, raw_data, metadata):
        """Generate response for transportation-related questions."""
        stops = raw_data.get("n_stops", "N/A")
        lines = raw_data.get("n_lines", "N/A")
        avg_distance = raw_data.get("avg_distan", "N/A")
        stop_coverage = raw_data.get("stop_coverage", 0)
        is_desert = metadata.get("is_transport_desert", False)
        
        response = f"In {year}, census area {census_id} ({zone}, District {district}) had {stops} public transport stops serving {lines} lines."
        
        if is_desert:
            response += " This area was classified as a 'transport desert' due to insufficient public transportation options."
        
        if avg_distance != -1 and avg_distance != "N/A":
            response += f" The average distance to the nearest stop was {avg_distance} meters."
        elif avg_distance == -1:
            response += " There was no data available on the average distance to stops, likely because there were no stops in this area."
        
        return response
    
    def _answer_accident_question(self, query, census_id, year, zone, district, raw_data, metadata):
        """Generate response for accident-related questions."""
        accidents = raw_data.get("tot_accidents", 0)
        vehicles = raw_data.get("tot_vehicles_involves", 0)
        public_vehicles = raw_data.get("n_public_vehicles", 0)
        has_accidents = metadata.get("has_accidents", False)
        public_involvement = metadata.get("public_vehicle_involvement", False)
        vehicle_categories = metadata.get("vehicle_categories", [])
        
        response = f"In {year}, census area {census_id} ({zone}, District {district}) had {accidents} reported traffic accidents"
        
        if vehicles != "N/A" and float(vehicles) > 0:
            response += f" involving a total of {vehicles} vehicles"
        
        response += "."
        
        if vehicle_categories and len(vehicle_categories) > 0:
            # Format vehicle types nicely
            if len(vehicle_categories) == 1:
                vehicle_types_text = vehicle_categories[0]
            elif len(vehicle_categories) == 2:
                vehicle_types_text = f"{vehicle_categories[0]} and {vehicle_categories[1]}"
            else:
                vehicle_types_text = ", ".join(vehicle_categories[:-1]) + ", and " + vehicle_categories[-1]
            
            response += f" The vehicles involved were categorized as: {vehicle_types_text}."
        
        if public_involvement:
            response += f" Public transportation vehicles were involved in some of these accidents."
        elif "public" in query or "bus" in query or "tram" in query:
            response += " No public transportation vehicles were involved in any of these accidents."
        
        return response
    
    def _answer_housing_question(self, query, census_id, year, zone, district, raw_data):
        """Generate response for housing-related questions."""
        families = raw_data.get("n_fam", "N/A")
        total_pop = raw_data.get("tot", "N/A")
        
        response = f"In {year}, census area {census_id} ({zone}, District {district}) had {families} households."
        
        # Calculate average family size if data is available
        if families != "N/A" and total_pop != "N/A" and float(families) > 0:
            avg_family_size = round(float(total_pop) / float(families), 2)
            response += f" The average household size was {avg_family_size} persons."
        
        # Add nearby area info if available
        families_500 = raw_data.get("fam_500", "N/A")
        if families_500 != "N/A":
            response += f" In the surrounding 500-meter radius, there were approximately {families_500} households."
        
        return response
    
    def _answer_general_question(self, query, census_id, year, zone, district, raw_data, metadata):
        """Generate response for general questions about the area."""
        area = raw_data.get("area", "N/A")
        total_pop = raw_data.get("tot", "N/A")
        stops = raw_data.get("n_stops", "N/A")
        lines = raw_data.get("n_lines", "N/A")
        accidents = raw_data.get("tot_accidents", "N/A")
        is_desert = metadata.get("is_transport_desert", False)
        
        response = f"In {year}, census area {census_id} ({zone}, District {district}) had a total population of {total_pop} residents"
        
        # Calculate population density if area available
        if area != "N/A" and total_pop != "N/A" and float(area) > 0:
            pop_density = round(float(total_pop) / (float(area)/10000), 2)
            response += f", with a population density of {pop_density} people per hectare"
        
        response += f". The area had {stops} public transport stops serving {lines} lines"
        
        if is_desert:
            response += " and was classified as a 'transport desert'"
        
        response += f". There were {accidents} traffic accidents reported in this area."
        
        return response

# Example usage
def main():
    # Change the file path to your JSON dataset
    rag = TurinCensusRAG("turin_rag_verbalizations.json")

    # Example questions to test with
    example_questions = [
        "How many vehicles were involved in accidents in Municipio in 2012?",
        "What was the population of census area 1 in 2012?",
        "How many public transit stops were in census area 1 in 2012?"
    ]

    for question in example_questions:
        print(f"\nQuestion: {question}")
        answer = rag.answer_question(question)
        print(f"Answer: {answer}")

    # Interactive mode
    print("\n\nTurin Census RAG Question Answering System")
    print("Type 'exit' to quit")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break
        answer = rag.answer_question(user_question)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
