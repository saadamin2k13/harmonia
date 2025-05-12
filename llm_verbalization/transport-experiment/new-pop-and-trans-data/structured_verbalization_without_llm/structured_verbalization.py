import pandas as pd
import json
from typing import Dict, Any

def generate_rag_verbalization(row: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a RAG-optimized verbalization with structured metadata and searchable text."""
    
    # Parse vehicle categories (clean and format)
    vehicle_categories = [
        v.strip().replace("_", " ") 
        for v in row['vehicle_categories']
    ] if isinstance(row['vehicle_categories'], list) else []
    
    # Core metadata (for filtering)
    metadata = {
        "census_id": row['cens'],
        "year": row['year'],
        "zone": row['desc_zone'],
        "district": row['district'],
        "has_transport": row['n_stops'] > 0,
        "has_accidents": row['tot_accidents'] > 0,
        "public_vehicle_involvement": row['n_public_vehicles'] > 0,
        "vehicle_categories": vehicle_categories,
        "is_transport_desert": row['avg_distan'] == -1,
    }
    
    # Natural language summary (for semantic search)
    text = (
        f"In {row['year']}, Census Area {row['cens']} ({row['desc_zone']}, District {row['district']}) "
        f"had {row['tot']} residents ({row['tot_F']} female, {row['tot_M']} male). "
        f"Transport: {row['n_stops']} stops, {row['n_lines']} lines. "
        f"Accidents: {row['tot_accidents']} involving {', '.join(vehicle_categories)}. "
        f"Key demographics: {row['minors']} minors, {row['seniors']} seniors. "
        f"Foreign population: {row['tot_foreig']}."
    )
    
    return {
        "id": f"{row['year']}-{row['cens']}",  # Unique ID for retrieval
        "metadata": metadata,
        "text": text,
        # Optional: Include raw fields for hybrid search
        "raw_data": {k: v for k, v in row.items() if k != 'vehicle_categories'}
    }

def process_csv(csv_path: str) -> list[Dict[str, Any]]:
    """Process CSV into RAG-ready documents."""
    df = pd.read_csv(csv_path)
    
    # Clean vehicle_categories (handle both strings and lists)
    df['vehicle_categories'] = df['vehicle_categories'].apply(
        lambda x: (
            x.strip("[]").replace("'", "").split(", ") 
            if isinstance(x, str) 
            else []
        )
    )
    
    return [generate_rag_verbalization(row) for _, row in df.iterrows()]

def main():
    verbalizations = process_csv("population-and-transport.csv")
    
    # Save for RAG pipeline
    with open("turin_rag_verbalizations.json", "w") as f:
        json.dump(verbalizations, f, indent=2)
    
    print(f"Generated {len(verbalizations)} RAG-optimized documents.")
    print("Sample document:")
    print(json.dumps(verbalizations[0], indent=2))

if __name__ == "__main__":
    main()
