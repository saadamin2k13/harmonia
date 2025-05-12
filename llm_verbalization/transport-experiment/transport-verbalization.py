import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Optional, Union, Any, Dict, List, Tuple
from enum import Enum

class AnalysisPerspective(Enum):
    POPULATION_COVERAGE = "population_coverage"
    DEMOGRAPHICS_ACCESSIBILITY = "demographics_accessibility"
    LINES_LENGTH = "lines_length"
    TEMPORAL_CHANGES = "temporal_changes"

class TransportVerbalizationAssistant:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: Optional[str] = None):
        """Initialize the verbalization assistant with a specified LLM."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self._initialize_model(model_name)
        self._initialize_perspective_prompts()

    def _initialize_model(self, model_name: str) -> None:
        """Initialize the model, tokenizer, and pipeline."""
        try:
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print(f"Loading model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            print("Creating text generation pipeline...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def _initialize_perspective_prompts(self) -> None:
        """Define prompt templates for each analysis perspective."""
        self.perspective_prompts = {
            AnalysisPerspective.POPULATION_COVERAGE: """
Analyze how public transport coverage relates to population size in this Turin district.

Key Metrics:
- Total Population: {pop}
- Stops per Capita: {perc_stops}
- Number of Lines with Stops: {n_lines_stopping}
- Number of Stops: {n_stops}

Additional Context:
- Census Zone: {sez_cens}
- Area: {area} square meters

Please analyze how the number of stops per capita and number of lines stopping reflect the population size in this district. Consider whether the transport coverage appears proportional to the population density. Use only the provided data without making assumptions.
""",
            
            AnalysisPerspective.DEMOGRAPHICS_ACCESSIBILITY: """
Analyze how public transport accessibility relates to vulnerable demographics in this Turin district.

Key Demographics:
- Percentage of Minors: {perc_minor}
- Percentage of Seniors: {perc_senior}
- Total Population: {pop}

Transport Accessibility:
- Number of Stops: {n_stops}
- Number of Lines with Stops: {n_lines_stopping}
- Stops per Capita: {perc_stops}

Please analyze whether this district's transport coverage appears appropriate for its demographic composition, particularly focusing on areas with higher percentages of minors and seniors. Use only the provided data without making assumptions.
""",
            
            AnalysisPerspective.LINES_LENGTH: """
Analyze the relationship between number of transport lines and their length in this Turin district.

Key Metrics:
- Number of Lines with Stops: {n_lines_stopping}
- Length of Lines Coverage per Capita: {perc_length_stopping}
- Total Population: {pop}
- Area: {area} square meters

Please analyze how the length of lines per capita relates to the number of lines in the area. Consider whether areas with more lines show proportionally greater coverage length. Use only the provided data without making assumptions.
""",

            AnalysisPerspective.TEMPORAL_CHANGES: """
Analyze the changes in public transportation and demographics between 2018 and 2019 for Census Zone {sez_cens}.

Changes in Demographics:
- Immigrant Percentage 2018: {per_immigrants_2018}
- Immigrant Percentage 2019: {per_immigrants_2019}
- Population 2018: {pop_2018}
- Population 2019: {pop_2019}

Changes in Transport:
- Stops 2018: {n_stops_2018}
- Stops 2019: {n_stops_2019}
- Lines 2018: {n_lines_stopping_2018}
- Lines 2019: {n_lines_stopping_2019}

Please analyze the temporal changes in this district, particularly focusing on:
1. How immigrant percentage has changed
2. Whether transport infrastructure (stops and lines) has been modified
3. How these changes might impact service delivery

Use only the provided data without making assumptions.
"""
        }

    def safe_get_value(
        self, 
        row: pd.Series, 
        column: str, 
        default: Any = None, 
        multiply_by_100: bool = False
    ) -> Union[str, Any]:
        """Safely get a value from a row with optional percentage conversion."""
        try:
            value = row.get(column, default)
            if pd.isna(value):
                return default
            if multiply_by_100 and isinstance(value, (int, float)):
                return f"{value * 100:.2f}%"
            return value
        except Exception:
            return default

    def _get_temporal_data(self, df: pd.DataFrame, census_zone: int) -> Dict[str, Any]:
        """Get temporal data for a specific census zone."""
        zone_data = df[df['sez_cens'] == census_zone].sort_values('year')
        if len(zone_data) != 2:
            raise ValueError(f"Expected 2 years of data for census zone {census_zone}")
            
        data_2018 = zone_data[zone_data['year'] == 2018].iloc[0]
        data_2019 = zone_data[zone_data['year'] == 2019].iloc[0]
        
        return {
            'sez_cens': census_zone,
            'per_immigrants_2018': self.safe_get_value(data_2018, 'per_immigrants', 'N/A', multiply_by_100=True),
            'per_immigrants_2019': self.safe_get_value(data_2019, 'per_immigrants', 'N/A', multiply_by_100=True),
            'pop_2018': self.safe_get_value(data_2018, 'pop', 'N/A'),
            'pop_2019': self.safe_get_value(data_2019, 'pop', 'N/A'),
            'n_stops_2018': self.safe_get_value(data_2018, 'n_stops', 'N/A'),
            'n_stops_2019': self.safe_get_value(data_2019, 'n_stops', 'N/A'),
            'n_lines_stopping_2018': self.safe_get_value(data_2018, 'n_lines_stopping', 'N/A'),
            'n_lines_stopping_2019': self.safe_get_value(data_2019, 'n_lines_stopping', 'N/A')
        }

    def create_verbalization_prompt(self, row: pd.Series, perspective: AnalysisPerspective) -> str:
        """Create a prompt for a specific analysis perspective."""
        # Prepare data dictionary with all possible fields
        data = {
            'pop': self.safe_get_value(row, 'pop', 'N/A'),
            'perc_stops': self.safe_get_value(row, 'perc_stops', 'N/A', multiply_by_100=True),
            'n_lines_stopping': self.safe_get_value(row, 'n_lines_stopping', 'N/A'),
            'n_stops': self.safe_get_value(row, 'n_stops', 'N/A'),
            'sez_cens': self.safe_get_value(row, 'sez_cens', 'N/A'),
            'area': self.safe_get_value(row, 'area', 'N/A'),
            'perc_minor': self.safe_get_value(row, 'perc_minor', 'N/A', multiply_by_100=True),
            'perc_senior': self.safe_get_value(row, 'perc_senior', 'N/A', multiply_by_100=True),
            'perc_length_stopping': self.safe_get_value(row, 'perc_length_stopping', 'N/A', multiply_by_100=True),
        }
        
        return self.perspective_prompts[perspective].format(**data)

    def _process_single_district(self, row: pd.Series, perspective: AnalysisPerspective) -> str:
        """Process a single district from a specific perspective."""
        try:
            prompt = self.create_verbalization_prompt(row, perspective)
            
            generated_texts = self.generator(
                prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.4,
                do_sample=True
            )
            
            verbalization = generated_texts[0]['generated_text'].replace(prompt, '').strip()
            
            return (f"\n{'='*80}\n"
                   f"District Analysis - Census Zone {self.safe_get_value(row, 'sez_cens')} "
                   f"({perspective.value})\n{'='*80}\n\n"
                   f"{verbalization}\n")
        except Exception as e:
            return f"Error processing district {row.get('sez_cens', 'Unknown')}: {e}\n"

    def _process_temporal_analysis(self, df: pd.DataFrame, census_zone: int) -> str:
        """Process temporal analysis for a specific census zone."""
        try:
            temporal_data = self._get_temporal_data(df, census_zone)
            prompt = self.perspective_prompts[AnalysisPerspective.TEMPORAL_CHANGES].format(**temporal_data)
            
            generated_texts = self.generator(
                prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.4,
                do_sample=True
            )
            
            verbalization = generated_texts[0]['generated_text'].replace(prompt, '').strip()
            
            return (f"\n{'='*80}\n"
                   f"Temporal Analysis - Census Zone {census_zone} (2018-2019)\n"
                   f"{'='*80}\n\n"
                   f"{verbalization}\n")
        except Exception as e:
            return f"Error processing temporal analysis for census zone {census_zone}: {e}\n"

    def _save_analyses(
        self,
        analyses: List[str],
        output_path: str,
        perspective: AnalysisPerspective
    ) -> None:
        """Save analyses to a file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"\nTURIN DISTRICT TRANSPORT ANALYSIS - {perspective.value}\n\n")
                for analysis in analyses:
                    f.write(analysis)
        except Exception as e:
            print(f"Error saving analyses: {e}")

    def verbalize_data(
        self,
        input_path: str,
        perspective: AnalysisPerspective,
        output_path: Optional[str] = None,
        batch_size: int = 5
    ) -> List[str]:
        """Verbalize the dataset from a specific perspective."""
        try:
            df = pd.read_excel(input_path)
            print(f"Processing data from {perspective.value} perspective")
            
            all_analyses = []
            
            if perspective == AnalysisPerspective.TEMPORAL_CHANGES:
                # Process temporal changes for specific census zones
                target_zones = [2535, 3712, 2756]
                for zone in target_zones:
                    analysis = self._process_temporal_analysis(df, zone)
                    all_analyses.append(analysis)
            else:
                # Process regular single-year analysis
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i + batch_size]
                    for _, row in batch.iterrows():
                        analysis = self._process_single_district(row, perspective)
                        all_analyses.append(analysis)
                        
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
            
            if output_path:
                self._save_analyses(all_analyses, output_path, perspective)
            
            return all_analyses
                        
        except Exception as e:
            raise RuntimeError(f"Failed to process data: {e}")

def main():
    """Main function to run the analysis."""
    try:
        print("Initializing Transport Verbalization Assistant...")
        assistant = TransportVerbalizationAssistant()

        # Define perspectives and their output files
        perspectives = [
            (AnalysisPerspective.POPULATION_COVERAGE, 'population_coverage_analysis.txt'),
            (AnalysisPerspective.DEMOGRAPHICS_ACCESSIBILITY, 'demographics_analysis.txt'),
            (AnalysisPerspective.LINES_LENGTH, 'lines_length_analysis.txt'),
            (AnalysisPerspective.TEMPORAL_CHANGES, 'temporal_changes_analysis.txt')
        ]

        # Process each perspective
        for perspective, output_file in perspectives:
            print(f"\nProcessing {perspective.value}...")
            assistant.verbalize_data(
                'df.xlsx',  # Update this path to your data file
                perspective,
                output_file
            )

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
