import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class SemanticParsingAssistant:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the semantic parsing assistant with a specified LLM.

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

    def create_drs_prompt(self, text):
        """
        Create a prompt for DRS parsing of input text, including examples.

        Args:
            text (str): Input text to be parsed

        Returns:
            str: Prompt for the LLM with examples
        """
        prompt = """Discourse Representation structure is a formal meaning representation in the form of first-order logic. It is a framework used in linguistics and natural language processing to represent the meaning of sentences. I will provide you with some examples of text-to-DRS parsing, and then you will generate DRS for new text.

For the given text "A room." the DRS representation is "event.v.01 Participant +1 room.n.01"

Similarly, for a given text "Liz Mohn" the DRS representation is "event.v.01 Participant +1 female.n.02 Name "Liz Mohn"".

Now, you will be given the text only and you have to provide with the DRS. 
Use examples above as a reference for you. 
Only generate one DRS per line based on the text given to you. 
You must generate DRS to the given text and it should not be left blank.

Text: """

        return f"{prompt}{text}\n\nDRS:"

    def parse_text(self, text):
        """
        Generate DRS representation for a single line of text.

        Args:
            text (str): Input text to parse

        Returns:
            str: DRS representation of the text
        """
        try:
            # Create the prompt
            prompt = self.create_drs_prompt(text)

            # Generate DRS using the pipeline
            print(f"Generating DRS for: {text}")
            generated_texts = self.generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.2,  # Even lower temperature for more consistent outputs
                do_sample=True
            )

            # Extract the generated text
            generated_text = generated_texts[0]['generated_text']
            drs = generated_text.replace(prompt, '').strip()

            return drs

        except Exception as e:
            print(f"Error generating DRS: {e}")
            return f"Could not generate DRS. Error: {e}"

    def process_text_file(self, input_path, output_path):
        """
        Process a text file line by line and generate DRS representations.

        Args:
            input_path (str): Path to the input text file
            output_path (str): Path to save the DRS representations

        Returns:
            None
        """
        try:
            # Read input file
            print(f"Reading text file from {input_path}...")
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            # Process each line
            print(f"Processing {len(lines)} lines...")
            results = []
            for i, line in enumerate(lines, 1):
                print(f"\nProcessing line {i}: {line}")
                drs = self.parse_text(line)
                results.append((line, drs))
                print(f"DRS generated for line {i}")

            # Save results
            print(f"Saving results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                for original, drs in results:
                    f.write(f"Text: {original}\n")
                    f.write(f"DRS: {drs}\n")
                    f.write("\n")  # Empty line between entries
            print("Results saved successfully!")

        except Exception as e:
            print(f"Error processing file: {e}")
            raise

def main():
    try:
        # Initialize the semantic parsing assistant
        print("Initializing Semantic Parsing Assistant...")
        parsing_assistant = SemanticParsingAssistant(
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )

        # Path to your input text file
        input_path = 'test_text.txt'

        # Path to save the output file
        output_path = 'drs_representations.txt'

        # Process the text file
        print("Starting text processing...")
        parsing_assistant.process_text_file(input_path, output_path)

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
