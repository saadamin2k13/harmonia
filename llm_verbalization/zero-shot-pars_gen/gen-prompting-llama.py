import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class DRSGenerationAssistant:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the DRS-to-text generation assistant with a specified LLM.

        Args:
            model_name (str): Hugging Face model name for the LLM
        """
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the tokenizer and model
        try:
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            print(f"Loading model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            # Create a text generation pipeline
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

    def create_generation_prompt(self, drs):
        """
        Create a prompt for text generation from DRS.

        Args:
            drs (str): Input DRS to be converted to text

        Returns:
            str: Prompt for the LLM with examples
        """
        prompt = """Generate natural language text from the following Discourse Representation Structure (DRS). Here are some examples:

Example 1:
DRS: event.v.01 Participant +1 room.n.01
Text: A room.

Example 2:
DRS: event.v.01 Participant +1 female.n.02 Name "Liz Mohn"
Text: Liz Mohn

Now, generate natural language text for the following DRS representation, following the same format as shown in the examples above:

DRS: """

        return f"{prompt}{drs}\n\nText:"

    def generate_text(self, drs):
        """
        Generate natural language text from a single DRS representation.

        Args:
            drs (str): Input DRS representation

        Returns:
            str: Generated natural language text
        """
        try:
            # Create the prompt
            prompt = self.create_generation_prompt(drs)

            # Generate text using the pipeline
            print(f"Generating text for DRS: {drs}")
            generated_texts = self.generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.2,  # Low temperature for consistent outputs
                do_sample=True
            )

            # Extract the generated text
            generated_text = generated_texts[0]['generated_text']
            text = generated_text.replace(prompt, '').strip()

            return text

        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Could not generate text. Error: {e}"

    def process_drs_file(self, input_path, output_path):
        """
        Process a file containing DRS representations line by line and generate text.

        Args:
            input_path (str): Path to the input DRS file
            output_path (str): Path to save the generated texts

        Returns:
            None
        """
        try:
            # Read input file
            print(f"Reading DRS file from {input_path}...")
            with open(input_path, 'r', encoding='utf-8') as f:
                drs_lines = [line.strip() for line in f if line.strip()]

            # Process each DRS
            print(f"Processing {len(drs_lines)} DRS representations...")
            results = []
            for i, drs in enumerate(drs_lines, 1):
                print(f"\nProcessing DRS {i}: {drs}")
                generated_text = self.generate_text(drs)
                results.append((drs, generated_text))
                print(f"Text generated for DRS {i}")

            # Save results
            print(f"Saving results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for drs, text in results:
                    f.write(f"DRS: {drs}\n")
                    f.write(f"Generated Text: {text}\n")
                    f.write("\n")  # Empty line between entries
            print("Results saved successfully!")

        except Exception as e:
            print(f"Error processing file: {e}")
            raise

def main():
    try:
        # Initialize the DRS generation assistant
        print("Initializing DRS-to-Text Generation Assistant...")
        generation_assistant = DRSGenerationAssistant(
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )

        # Path to your input DRS file
        input_path = 'test_sbn.txt'

        # Path to save the output file
        output_path = 'generated_texts.txt'

        # Process the DRS file
        print("Starting DRS processing...")
        generation_assistant.process_drs_file(input_path, output_path)

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
