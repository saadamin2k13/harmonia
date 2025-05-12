import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class TransportVerbalizationAssistant:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the verbalization assistant with a specified LLM.

        Args:
            model_name (str): Hugging Face model name for the LLM
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

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
                tokenizer=self.tokenizer
            )

            print("Model and pipeline initialized successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def safe_get_value(self, row, column, default=None, multiply_by_100=False):
        """
        Safely get a value from a row, with optional multiplication by 100 for percentages.
        
        Args:
            row (pd.Series): Data row
            column (str): Column name
            default: Default value if column doesn't exist
            multiply_by_100 (bool): Whether to multiply the value by 100
            
        Returns:
            Formatted value or default value
        """
        try:
            value = row.get(column, default)
            if pd.isna(value):
                return default
            if multiply_by_100:
                return f"{value * 100:.2f}%"
            return value
        except Exception:
            return default

    def create_verbalization_prompt(self, row):
        """
        Create a prompt to verbalize a single row of data focusing on multiple
        perspectives of transport coverage and demographics.

        Args:
            row (pd.Series): A single row from the DataFrame

        Returns:
            str: A prompt for analyzing transport coverage and population
        """
        prompt = f"""Analyze the following demographic and public transport data for a district in Turin, focusing on three key aspects: the relationship between population size and transport coverage, the connection between demographic groups (especially minors and seniors) and transport accessibility, and the correlation between number of transport lines and their total length per capita. Provide the analysis in flowing paragraphs that naturally connect these aspects.

District Details:
Year: {self.safe_get_value(row, 'year', 'N/A')}
Census Zone: {self.safe_get_value(row, 'sez_cens', 'N/A')}
Statistical Zone: {self.safe_get_value(row, 'stat_zone', 'N/A')}
Area: {self.safe_get_value(row, 'area', 'N/A')} square meters

Population Demographics:
- Total Population: {self.safe_get_value(row, 'pop', 'N/A')}
- Female Percentage: {self.safe_get_value(row, 'perc_f', 'N/A', multiply_by_100=True)}
- Immigrant Percentage: {self.safe_get_value(row, 'per_immigrants', 'N/A', multiply_by_100=True)}
- Female Immigrants: {self.safe_get_value(row, 'perc_immigrants_F', 'N/A', multiply_by_100=True)}
- Minors Percentage: {self.safe_get_value(row, 'perc_minor', 'N/A', multiply_by_100=True)}
- Senior Citizens: {self.safe_get_value(row, 'perc_senior', 'N/A', multiply_by_100=True)}

Public Transport Coverage:
- Number of Stops: {self.safe_get_value(row, 'n_stops', 'N/A')}
- Number of Lines with Stops: {self.safe_get_value(row, 'n_lines_stopping', 'N/A')}
- Stops per Capita: {self.safe_get_value(row, 'perc_stops', 'N/A', multiply_by_100=True)}
- Stops per Line: {self.safe_get_value(row, 'perc_stops_per_line_stopping', 'N/A', multiply_by_100=True)}
- Length of Lines Coverage: {self.safe_get_value(row, 'perc_length_stopping', 'N/A', multiply_by_100=True)}

Please provide a flowing analysis that addresses:
1. How the number of stops per capita and number of lines stopping reflect the population size
2. Whether areas with higher percentages of minors and seniors show corresponding levels of transport coverage
3. How the length of lines per capita relates to the number of lines in the area

Write your analysis in natural, connected paragraphs that smoothly transition between these aspects. Use only the information provided above without making assumptions:"""

        return prompt

    def verbalize_row(self, row):
        """
        Generate a verbal description for a single row using the LLM.

        Args:
            row (pd.Series): A single row from the DataFrame

        Returns:
            str: Verbalized analysis of transport and demographic data
        """
        try:
            prompt = self.create_verbalization_prompt(row)

            print("Generating analysis...")
            generated_texts = self.generator(
                prompt,
                max_length=1000,
                num_return_sequences=1,
                temperature=0.4,
                do_sample=True
            )

            generated_text = generated_texts[0]['generated_text']
            verbalization = generated_text.replace(prompt, '').strip()

            return verbalization

        except Exception as e:
            print(f"Error generating analysis: {e}")
            return f"Could not generate analysis. Error: {str(e)}"

    def verbalize_data(self, input_path, output_path):
        """
        Verbalize the entire dataset and save results to a text file.

        Args:
            input_path (str): Path to the input Excel file
            output_path (str): Path to save the verbalized descriptions

        Returns:
            None
        """
        print(f"Reading Excel file from {input_path}...")
        try:
            df = pd.read_excel(input_path)
            print(f"Columns in dataset: {', '.join(df.columns)}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return

        print(f"Verbalizing {len(df)} districts...")
        all_verbalizations = []
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"\nProcessing District {i} (Census Zone {row['sez_cens']}):")
            verbalization = self.verbalize_row(row)
            
            formatted_text = (f"\n{'='*80}\n"
                            f"District Analysis - Census Zone {self.safe_get_value(row, 'sez_cens')} "
                            f"(Year {self.safe_get_value(row, 'year')})\n{'='*80}\n\n"
                            f"{verbalization}\n")
            
            all_verbalizations.append(formatted_text)
            print(f"Analysis generated for District {i}")

        print(f"Saving analyses to {output_path}...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\nTURIN DISTRICT TRANSPORT AND DEMOGRAPHICS ANALYSIS\n\n")
                for text in all_verbalizations:
                    f.write(text)
            print("Analyses saved successfully!")
        except Exception as e:
            print(f"Error saving output file: {e}")


def main():
    try:
        print("Initializing Transport Verbalization Assistant...")
        verbalization_assistant = TransportVerbalizationAssistant(
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )

        # Update these paths according to your file locations
        input_path = 'df.xlsx'
        output_path = 'transport_analysis.txt'

        print("Starting Data Verbalization...")
        verbalization_assistant.verbalize_data(input_path, output_path)

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
