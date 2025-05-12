import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import os
from tqdm import tqdm
import logging
import time
from typing import Dict, Any, Optional, List, Union
import argparse

class UrbanDataVerbalizer:
    """A class to convert urban census data into human-readable narratives."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        output_dir: str = "verbalization_results",
        batch_size: int = 50,
        max_new_tokens: int = 512,  # Changed from max_length to max_new_tokens
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        log_level: str = "INFO"
    ):
        """
        Initialize the UrbanDataVerbalizer with model and processing parameters.
        
        Args:
            model_name: Hugging Face model identifier
            output_dir: Directory to save output files
            batch_size: Number of rows to process in one batch
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition in generated text
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Set up logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Store generation parameters
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens  # Store as max_new_tokens instead of max_length
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # Initialize model and tokenizer
        self.logger.info(f"Initializing model: {model_name}")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate dtype based on device
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto"
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            self.logger.info("Model initialization successful")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _setup_logging(self, log_level: str) -> None:
        """Set up logging with appropriate format and level."""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _format_row_for_prompt(self, row: pd.Series) -> Dict[str, Any]:
        """
        Format a dataframe row into a cleaner dictionary for the prompt.
        Handles different data types and formats appropriately.
        
        Args:
            row: A pandas Series containing one row of data
            
        Returns:
            Dictionary with selected and formatted row data
        """
        # Extract a reasonable subset of keys for the verbalization
        # Include the most relevant fields for a meaningful narrative
        try:
            target_input = {
                "year": int(row["year"]),
                "cens": int(row["cens"]),
                "zone_stat": int(row["zone_stat"]),
                "desc_zone": row["desc_zone"],
                "district": int(row["district"]),
                "area": float(row["area"]),
                "n_stops": int(row["n_stops"]),
                "n_lines": int(row["n_lines"]),
                "avg_distan": float(row["avg_distan"]) if row["avg_distan"] != -1 else "no public transport",
                "tot": int(row["tot"]),
                "tot_F": int(row["tot_F"]),
                "tot_M": int(row["tot_M"]),
                "tot_foreig": int(row["tot_foreig"]),
                "minors": int(row["minors"]),
                "working_age": int(row["working_age"]),
                "seniors": int(row["seniors"]),
                "stop_coverage": float(row["stop_coverage"]),
                "tot_accidents": int(row["tot_accidents"]),
                "tot_vehicles_involves": int(row["tot_vehicles_involves"]) if "tot_vehicles_involves" in row else 0,
                "vehicle_categories": self._parse_vehicle_categories(row["vehicle_categories"]),
                "n_public_vehicles": int(row["n_public_vehicles"])
            }
            
            # Add public_accidents if available
            if "public_accidents" in row:
                target_input["public_accidents"] = int(row["public_accidents"])
                
            return target_input
            
        except Exception as e:
            self.logger.error(f"Error formatting row: {e}")
            # Return a minimal version to avoid breaking the process
            return {
                "year": int(row.get("year", 0)),
                "cens": int(row.get("cens", 0)),
                "desc_zone": row.get("desc_zone", "Unknown"),
                "tot": int(row.get("tot", 0))
            }
    
    def _parse_vehicle_categories(self, categories_str: str) -> List[str]:
        """
        Parse the vehicle categories string from the dataframe into a clean list.
        
        Args:
            categories_str: String representation of the vehicle categories list
            
        Returns:
            List of vehicle category strings
        """
        try:
            # Handle various string formats for the vehicle categories
            if isinstance(categories_str, str):
                # Remove brackets and quotes and split by comma
                cleaned = categories_str.strip('[]\'\"')
                if not cleaned:  # Empty string
                    return []
                    
                # Try to parse as a list-like string
                if cleaned.startswith('"') or cleaned.startswith("'"):
                    # Looks like a JSON format
                    try:
                        return json.loads(categories_str.replace("'", '"'))
                    except json.JSONDecodeError:
                        pass
                
                # Split by commas and clean up each item
                items = cleaned.split(',')
                return [item.strip().strip('\'"') for item in items if item.strip()]
            elif isinstance(categories_str, list):
                return categories_str
            else:
                self.logger.warning(f"Unexpected vehicle categories type: {type(categories_str)}")
                return []
        except Exception as e:
            self.logger.error(f"Error parsing vehicle categories: {e}")
            return []
    
    def generate_comprehensive_narrative(self, row: pd.Series) -> str:
        """
        Generate a natural language narrative for a row of urban data using few-shot prompting.
        
        Args:
            row: A pandas Series containing one row of urban census data
            
        Returns:
            A generated narrative description of the data
        """
        start_time = time.time()
        try:
            # Format the row data for the prompt
            target_input = self._format_row_for_prompt(row)
            
            # Use a more concise prompt with fewer examples to manage token count
            # This helps prevent the error with token limits
            few_shot_prompt = """
You are an expert urban data analyst. Convert census and transport data into clear narratives.

Example:
Input:
{
  "year": 2012,
  "cens": 1,
  "zone_stat": 1,
  "desc_zone": "Municipio",
  "district": 1,
  "area": 13878.5615,
  "n_stops": 0,
  "n_lines": 0,
  "avg_distan": -1,
  "tot": 333,
  "tot_F": 197,
  "tot_M": 136,
  "tot_foreig": 114,
  "n_fam": 238,
  "minors": 33,
  "working_age": 226,
  "seniors": 74,
  "tot_500": 6580,
  "F_500": 3671,
  "tot_M_500": 3241,
  "foreign_500": 2228,
  "fam_500": 4471,
  "minors_500": 1122,
  "working_500": 4237,
  "seniors_500": 1553,
  "stop_coverage": 0,
  "tot_accidents": 2,
  "tot_vehicles_involves": 4,
  "vehicle_categories": ["veicoli_grandi", "privato_moto", "privato_vettura", "altro"],
  "n_public_vehicles": 0,
  "public_accidents": 0
}
Output:
In 2012, census area 1 of Turin with the statistical zone 1 (Municipio) in district 1 covered an area of approximately 13,879 square meters. It had a small population of 333 residents, with 197 females and 136 males, including 114 foreign nationals. There were 238 families, and the demographic distribution included 33 minors, 226 working-age adults, and 74 seniors. The extended population within a 500-meter radius reached 6,580 people, comprising 3,671 females, 3,241 males, 2,228 foreigners, and 4,471 families. Within this radius, there were 1,122 minors, 4,237 working-age individuals, and 1,553 seniors. Despite this population density, there were no public transport stops or lines in the zone, resulting in zero coverage and no transfer distance reported. In terms of road safety, there were 2 accidents involving 4 vehicles, including large vehicles, private motorcycles, private cars, and other categories. No public transport vehicles were involved in any accidents.

Now create a detailed narrative about:
Input:
""" + json.dumps(target_input, ensure_ascii=False) + "\nOutput:"

            # Calculate token count for logging
            input_tokens = len(self.tokenizer.encode(few_shot_prompt))
            self.logger.debug(f"Input prompt token count: {input_tokens}")
            
            # Generate the narrative using max_new_tokens instead of max_length
            response = self.generator(
                few_shot_prompt,
                max_new_tokens=self.max_new_tokens,  # Use max_new_tokens instead of max_length
                num_return_sequences=1,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=self.repetition_penalty
            )

            # Extract the generated narrative from the response
            generated_text = response[0]['generated_text']
            if "Output:" in generated_text:
                narrative = generated_text.split("Output:")[-1].strip()
            else:
                narrative = generated_text.strip()

            # Log processing time for monitoring
            processing_time = time.time() - start_time
            self.logger.debug(f"Narrative generated in {processing_time:.2f} seconds")
            
            return narrative

        except Exception as e:
            self.logger.error(f"Error generating narrative: {e}")
            return f"Error processing data: {str(e)}"

    def verbalize_dataset(
        self, 
        input_file: str, 
        output_file: Optional[str] = None,
        skip_rows: int = 0,
        max_rows: Optional[int] = None
    ) -> str:
        """
        Process an entire dataset and generate narratives for each row.
        
        Args:
            input_file: Path to the CSV file containing urban data
            output_file: Name of output file (defaults to input filename with .txt extension)
            skip_rows: Number of rows to skip from the beginning
            max_rows: Maximum number of rows to process (None for all)
            
        Returns:
            Path to the output file
        """
        try:
            # Set default output filename if not provided
            if output_file is None:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                output_file = f"{base_name}_verbalized.txt"
            
            # Construct full output path
            output_path = os.path.join(self.output_dir, output_file)
            
            # Load and prepare the dataset
            self.logger.info(f"Loading dataset from {input_file}")
            df = pd.read_csv(input_file)
            self.logger.info(f"Dataset loaded with {len(df)} rows")
            
            # Apply row limits if specified
            if skip_rows > 0:
                df = df.iloc[skip_rows:]
                self.logger.info(f"Skipped {skip_rows} rows")
            
            if max_rows is not None:
                df = df.iloc[:max_rows]
                self.logger.info(f"Limited to {max_rows} rows")
            
            total_rows = len(df)
            
            # Process the dataset in batches
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(f"# Urban Data Verbalization Results\n")
                outfile.write(f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                outfile.write(f"# Source file: {input_file}\n")
                outfile.write(f"# Total records: {total_rows}\n\n")
                
                with tqdm(total=total_rows, desc="Verbalizing Urban Data") as pbar:
                    for i in range(0, total_rows, self.batch_size):
                        batch = df.iloc[i:min(i+self.batch_size, total_rows)]
                        
                        for idx, row in batch.iterrows():
                            try:
                                # Generate narrative for the current row
                                narrative = self.generate_comprehensive_narrative(row)
                                
                                # Write to output file with metadata
                                record_id = f"Census {row['cens']}, Zone {row['zone_stat']}, Year {row['year']}"
                                outfile.write(f"## {record_id}\n\n")
                                outfile.write(narrative + "\n\n")
                                outfile.write("-" * 80 + "\n\n")
                            except Exception as e:
                                # Log error but continue processing other rows
                                self.logger.error(f"Error processing row {idx}: {e}")
                                outfile.write(f"## Census {row.get('cens', 'Unknown')}, Error Processing\n\n")
                                outfile.write(f"Error: {str(e)}\n\n")
                                outfile.write("-" * 80 + "\n\n")
                            finally:
                                # Update progress bar even if there was an error
                                pbar.update(1)
            
            self.logger.info(f"Verbalization complete. Results saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error verbalizing dataset: {e}")
            raise

    def verbalize_single_record(self, record_data: Union[Dict, pd.Series]) -> str:
        """
        Generate a narrative for a single record provided as a dictionary or Series.
        
        Args:
            record_data: Dictionary or Series containing urban data fields
            
        Returns:
            Generated narrative
        """
        if isinstance(record_data, dict):
            # Convert dict to pandas Series for consistency
            record_series = pd.Series(record_data)
        else:
            record_series = record_data
            
        return self.generate_comprehensive_narrative(record_series)


def main():
    """Command-line interface for the Urban Data Verbalizer."""
    parser = argparse.ArgumentParser(description="Verbalize urban census and transport data")
    
    # Input and output
    parser.add_argument("input_file", help="Path to the CSV file containing urban data")
    parser.add_argument("--output-file", help="Name of output file")
    parser.add_argument("--output-dir", default="verbalization_results", help="Directory to save output files")
    
    # Model configuration
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", help="Hugging Face model identifier")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of rows to process in one batch")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repetition")
    
    # Processing limits
    parser.add_argument("--skip-rows", type=int, default=0, help="Number of rows to skip")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")
    
    # Other options
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Initialize and run the verbalizer
    verbalizator = UrbanDataVerbalizer(
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,  # Changed parameter name
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        log_level=args.log_level
    )
    
    output_path = verbalizator.verbalize_dataset(
        args.input_file,
        args.output_file,
        skip_rows=args.skip_rows,
        max_rows=args.max_rows
    )
    
    print(f"Verbalization complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
