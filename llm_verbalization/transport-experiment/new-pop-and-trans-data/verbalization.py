import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Optional, Union, Any, Dict, List
from enum import Enum

class AnalysisPerspective(Enum):
    TRANSPORT_ACCESSIBILITY = "transport_accessibility"
    DEMOGRAPHICS_MOBILITY = "demographics_mobility"
    TRAFFIC_SAFETY = "traffic_safety"
    TEMPORAL_TRENDS = "temporal_trends"

class VerbalizationPipeline:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model(model_name)
        self._initialize_prompts()

    def _initialize_model(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, device_map="auto"
        )
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")

    def _initialize_prompts(self):
        self.prompts = {
            AnalysisPerspective.TRANSPORT_ACCESSIBILITY: """
Analyze the public transport accessibility in Census Zone {cens}, District {district} (Year {year}).
- Number of stops: {n_stops}
- Number of transport lines: {n_lines}
- Stop coverage ratio: {stop_coverage}
- Average transport connectivity distance: {avg_distan}
Provide an analytical assessment of how well this area is connected by public transport.
""",
            AnalysisPerspective.DEMOGRAPHICS_MOBILITY: """
Analyze the mobility patterns in Census Zone {cens}, District {district} (Year {year}).
- Total population: {tot}
- Gender Breakdown: Female: {tot_F}, Male: {tot_M}
- Minors: {minors}, Seniors: {seniors}, Working-age: {working_age}
- Foreign population: {tot_foreig}
- Number of families: {n_fam}
Evaluate whether public transport meets the mobility needs of different groups.
""",
            AnalysisPerspective.TRAFFIC_SAFETY: """
Analyze the traffic safety in Census Zone {cens}, District {district} (Year {year}).
- Total accidents: {tot_accidents}
- Public transport accidents: {public_accidents}
- Vehicles involved: {tot_vehicles_involves}
- Categories of involved vehicles: {vehicle_categories}
Provide insights on traffic risks and accident trends in this area.
""",
            AnalysisPerspective.TEMPORAL_TRENDS: """
Analyze the transport and demographic changes in Census Zone {cens}, District {district} from 2012 to 2019.
{n_stops_trend_section}{pop_trend_section}{accident_trend_section}
Summarize the key changes over time and their possible impact.
"""
        }
    
    def safe_get_value(self, row: pd.Series, column: str, default: Any = "N/A") -> Union[str, Any]:
        return row[column] if column in row and pd.notna(row[column]) else default
    
    def generate_text(self, row: pd.Series, perspective: AnalysisPerspective) -> str:
        formatted_data = {key: self.safe_get_value(row, key) for key in row.index}
        prompt_template = self.prompts[perspective]
        
        # Dynamically create missing sections for temporal trends
        formatted_data["n_stops_trend_section"] = f"- Transport stops change: {formatted_data['n_stops_trend']}\n" if "n_stops_trend" in formatted_data else ""
        formatted_data["pop_trend_section"] = f"- Population change: {formatted_data['pop_trend']} (Total: {formatted_data['tot']}, Foreign: {formatted_data['tot_foreig']})\n" if "pop_trend" in formatted_data else ""
        formatted_data["accident_trend_section"] = f"- Accident trends: {formatted_data['accident_trend']}\n" if "accident_trend" in formatted_data else ""
        
        prompt = prompt_template.format(**formatted_data)
        
        response = self.generator(
            prompt, max_length=500, num_return_sequences=1, temperature=0.5, do_sample=True, truncation=True
        )
        return response[0]['generated_text'].replace(prompt, '').strip()
    
    def process_data(self, df: pd.DataFrame) -> List[str]:
        results = []
        for _, row in df.iterrows():
            for perspective in AnalysisPerspective:
                text = self.generate_text(row, perspective)
                results.append(f"Census Zone {row['cens']}, District {row['district']}, Year {row['year']} - {perspective.value}:\n{text}\n\n")
        return results
    
    def save_results_to_text(self, results: List[str], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(results)
        print(f"Verbalized data saved to {output_path}")
    
# Usage Example
if __name__ == "__main__":
    df = pd.read_csv("pop-and-trans-data.csv")
    pipeline = VerbalizationPipeline()
    verbalized_results = pipeline.process_data(df)
    pipeline.save_results_to_text(verbalized_results, "verbalized_output.txt")

