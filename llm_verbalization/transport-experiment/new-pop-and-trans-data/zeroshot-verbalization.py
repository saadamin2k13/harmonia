import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import os
from tqdm import tqdm

class UrbanDataVerbalizer:
    def __init__(self,
                 model_name="meta-llama/Llama-2-7b-chat-hf",
                 output_dir="verbalization_results",
                 batch_size=50):
        """
        Initialize the verbalization pipeline with enhanced configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Increased sequence length for comprehensive narratives
        self.max_length = 2048  # Increased from previous implementation

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        # Updated generation pipeline with explicit truncation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def prepare_row_context(self, row):
        """
        Prepare contextual identifiers for the row

        Args:
            row (pd.Series): DataFrame row

        Returns:
            str: Formatted context string
        """
        return (f"Year {row['year']}, Census Area {row['cens']}, "
                f"Statistical Zone {row['zone_stat']} ({row['desc_zone']}), "
                f"District {row['district']}: ")

    def generate_comprehensive_narrative(self, row):
        """
        Generate a single-paragraph comprehensive narrative

        Args:
            row (pd.Series): DataFrame row with urban data

        Returns:
            str: Comprehensive verbalization of the row
        """
        try:
            # Prepare row context and facts
            row_context = self.prepare_row_context(row)
            facts = {key: row.get(key, 'Not available') for key in row.index}

            # Construct detailed prompt for single-paragraph narrative
            prompt = f"""Generate a comprehensive, single-paragraph narrative about an urban area
            based on the following numeric data. Ensure the narrative is concise, informative,
            and covers all key aspects of the urban landscape while maintaining
            the accuracy of the following facts:

            Unique Identifier: {row_context}
            Numeric Facts: {json.dumps(facts)}

            Generated Narrative:"""

            # Generate narrative with comprehensive settings
            response = self.generator(
                prompt,
                max_length=self.max_length,  # Increased sequence length
                num_return_sequences=1,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                truncation=True,  # Explicitly activate truncation
                repetition_penalty=1.2
            )

            # Extract and combine context with generated text
            narrative = response[0]['generated_text'].replace(prompt, '').strip()
            return row_context + narrative

        except Exception as e:
            print(f"Error generating narrative: {e}")
            return f"Error processing row: {str(e)}"

    def verbalize_dataset(self, input_csv, output_file='verbalized_urban_data.txt'):
        """
        Verbalize entire dataset with progress tracking
        """
        df = pd.read_csv(input_csv)
        total_rows = len(df)

        output_path = os.path.join(self.output_dir, output_file)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            with tqdm(total=total_rows, desc="Verbalizing Urban Data") as pbar:
                for i in range(0, total_rows, self.batch_size):
                    batch = df.iloc[i:i+self.batch_size]

                    for _, row in batch.iterrows():
                        # Generate narrative
                        narrative = self.generate_comprehensive_narrative(row)

                        # Write narrative with separator
                        outfile.write(narrative + "\n")
                        outfile.write("-" * 80 + "\n")  # Separator

                        # Update progress bar
                        pbar.update(1)

        print(f"Verbalization complete. Results saved to {output_path}")

# Usage Example
if __name__ == "__main__":
    verbalizator = UrbanDataVerbalizer(batch_size=50)
    verbalizator.verbalize_dataset("population-and-transport.csv")
