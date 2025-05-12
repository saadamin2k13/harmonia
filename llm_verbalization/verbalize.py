import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class CSVVerbalizationAssistant:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the verbalization assistant with a specified LLM.

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

    def create_verbalization_prompt(self, row):
        """
        Create a prompt to verbalize a single row of data.

        Args:
            row (pd.Series): A single row from the DataFrame

        Returns:
            str: A detailed textual description of the property
        """
        # Construct a detailed prompt for the LLM
        prompt = f"""Convert the following Airbnb property details into a descriptive paragraph. Only use the facts provided; do not add information that is not present in the data.
        Create a property description emphasizing its affordability and value for money. Highlight how this property offers a comfortable stay within a reasonable budget:

Property Details:
- Property Type: {row['Property Type']}
- Location: {row['City']}, {row['Country']} (Coordinates: {row['Latitude']}, {row['Longitude']})
- Bedrooms: {row['Bedrooms']}
- Bathrooms: {row['Bathrooms']}
- Maximum Guests: {row['Max Guests']}

Pricing:
- Monthly Rate: ${row['Published Monthly Rate (USD)']}
- Weekly Rate: ${row['Published Weekly Rate (USD)']}
- Cleaning Fee: ${row['Cleaning Fee (USD)']}

Booking Details:
- Check-in: {row['Check-in Time']}
- Check-out: {row['Checkout Time']}
- Minimum Stay: {row['Minimum Stay']} nights
- Instant Booking: {'Enabled' if row['Instantbook Enabled'] else 'Disabled'}

Ratings:
- Overall Rating: {row['Overall Rating']}
- Number of Reviews: {row['Number of Reviews']}

Provide a concise and compelling description based only on the information provided above. Focus on presenting the property as an economical option for budget-conscious travelers:"""

        return prompt

    def verbalize_row(self, row):
        """
        Generate a verbal description for a single row using the LLM.

        Args:
            row (pd.Series): A single row from the DataFrame

        Returns:
            str: Verbalized description of the property
        """
        try:
            # Create the prompt
            prompt = self.create_verbalization_prompt(row)

            # Generate text using the pipeline
            print("Generating description...")
            generated_texts = self.generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.5,  # Lower temperature to reduce hallucinations
                do_sample=True
            )

            # Extract the generated text
            generated_text = generated_texts[0]['generated_text']
            verbalization = generated_text.replace(prompt, '').strip()

            return verbalization

        except Exception as e:
            print(f"Error generating description: {e}")
            return f"Could not generate description. Error: {e}"

    def verbalize_csv(self, csv_path, output_path):
        """
        Verbalize an entire CSV file of properties.

        Args:
            csv_path (str): Path to the input CSV file
            output_path (str): Path to save the verbalized descriptions

        Returns:
            None
        """
        # Read the CSV file
        print(f"Reading CSV file from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Verbalize each row
        print(f"Verbalizing {len(df)} properties...")
        verbalizations = []
        for i, (_, row) in enumerate(df.iterrows(), 1):
            print(f"\nProcessing Property {i}:")
            verbalization = self.verbalize_row(row)
            verbalizations.append(verbalization)
            print(f"Description generated for Property {i}")

        # Save the results to a new CSV file
        print(f"Saving verbalizations to {output_path}...")
        df['Description'] = verbalizations
        df.to_csv(output_path, index=False)
        print("Verbalizations saved successfully!")


def main():
    try:
        # Initialize the verbalization assistant
        print("Initializing Verbalization Assistant...")
        verbalization_assistant = CSVVerbalizationAssistant(
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )

        # Path to your input CSV file
        csv_path = 'airbnb_turin_properties.csv'

        # Path to save the output CSV file
        output_path = 'property_description.csv'

        # Verbalize the CSV
        print("Starting CSV Verbalization...")
        verbalization_assistant.verbalize_csv(csv_path, output_path)

    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
