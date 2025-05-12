import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime

class PropertyMarketAnalyzer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the market analyzer with a specified LLM.
        
        Args:
            model_name (str): Hugging Face model name for the LLM
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            print(f"Loading tokenizer and model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print("Model initialization complete!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_and_process_data(self, airbnb_path, idealista_path):
        """
        Load and process data from both Airbnb and Idealista sources.
        
        Args:
            airbnb_path (str): Path to Airbnb CSV file
            idealista_path (str): Path to Idealista CSV file
            
        Returns:
            tuple: Processed DataFrames for both sources
        """
        try:
            # Load Airbnb data
            airbnb_df = pd.read_csv(airbnb_path)
            # Fill NaN values in Neighborhood with "Unknown"
            airbnb_df['Neighborhood'] = airbnb_df['City'].fillna("Unknown")
            
            # Load Idealista data
            idealista_df = pd.read_csv(idealista_path)
            # Process Idealista data to extract neighborhood information
            idealista_df['Neighborhood'] = idealista_df['APROX LOCATIONNAME'].fillna("Unknown")
            
            return airbnb_df, idealista_df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_market_analysis_prompt(self, airbnb_row, idealista_data):
        """
        Create a comprehensive prompt that combines short-term and long-term market data.
        
        Args:
            airbnb_row (pd.Series): A single row from Airbnb data
            idealista_data (pd.DataFrame): Idealista data for matching
            
        Returns:
            str: A detailed prompt for market analysis
        """
        # Safely get neighborhood
        neighborhood = str(airbnb_row['Neighborhood'])
        
        # Get matching Idealista data for the neighborhood
        matching_idealista = None
        try:
            matching_data = idealista_data[
                idealista_data['Neighborhood'].str.lower() == neighborhood.lower()
            ]
            if not matching_data.empty:
                matching_idealista = matching_data.iloc[0]
        except Exception as e:
            print(f"Error matching neighborhood data: {e}")

        prompt = f"""Analyze the following property market data and provide insights for policy considerations. Include both short-term rental and long-term market dynamics.

Short-Term Rental Property (Airbnb):
- Property Type: {airbnb_row['Property Type']}
- Location: {airbnb_row['City']}, {airbnb_row['Country']}
- Size: {airbnb_row['Bedrooms']} bedrooms, {airbnb_row['Bathrooms']} bathrooms
- Capacity: {airbnb_row['Max Guests']} maximum guests
- Short-term Pricing:
  * Monthly Rate: ${airbnb_row['Published Monthly Rate (USD)']}
  * Weekly Rate: ${airbnb_row['Published Weekly Rate (USD)']}
- Performance Metrics:
  * Rating: {airbnb_row['Overall Rating']}
  * Reviews: {airbnb_row['Number of Reviews']}
"""

        if matching_idealista is not None:
            prompt += f"""
Long-Term Market Data (Neighborhood: {neighborhood}):
- Residential Sale Prices (per sqm):
  * Average: €{matching_idealista['UNITPRICE_RESIDENTIAL_SALE_ALL']}
  * 1-bedroom: €{matching_idealista['UNITPRICE_HOME_SALE_ALL']}
  * 2-bedroom: €{matching_idealista['UNITPRICE_CHALET_SALE_ALL']}
- Long-term Rental Prices (per sqm):
  * Average: €{matching_idealista['UNITPRICE_RESIDENTIAL_RENT_ALL']}
  * 1-bedroom: €{matching_idealista['UNITPRICE_HOME_RENT_ALL']}
  * 2-bedroom: €{matching_idealista['UNITPRICE_CHALET_RENT_ALL']}
"""
        else:
            prompt += f"\nNote: No long-term market data available for neighborhood: {neighborhood}\n"

        prompt += """
Please provide a comprehensive analysis that includes:
Market dynamics comparison between short-term and long-term rentals, potential impact on housing affordability, suggestions for balanced policy approaches, and recommendations for sustainable tourism and housing market balance.

Generate a detailed response based only on the provided data:"""

        return prompt

    def generate_market_analysis(self, airbnb_row, idealista_data):
        """
        Generate a comprehensive market analysis using the LLM.
        
        Args:
            airbnb_row (pd.Series): A single row from Airbnb data
            idealista_data (pd.DataFrame): Idealista data for matching neighborhood
            
        Returns:
            str: Generated market analysis
        """
        try:
            prompt = self.create_market_analysis_prompt(airbnb_row, idealista_data)
            
            generated_texts = self.generator(
                prompt,
                max_length=900,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            analysis = generated_texts[0]['generated_text'].replace(prompt, '').strip()
            return analysis
        except Exception as e:
            print(f"Error generating analysis: {e}")
            return f"Could not generate analysis. Error: {e}"

    def analyze_market(self, airbnb_path, idealista_path, output_csv_path, output_text_path):
        """
        Analyze the market and save results to both CSV and text files.
        
        Args:
            airbnb_path (str): Path to Airbnb CSV file
            idealista_path (str): Path to Idealista CSV file
            output_csv_path (str): Path to save the CSV results
            output_text_path (str): Path to save the text analyses
        """
        print("Starting market analysis...")
        airbnb_df, idealista_df = self.load_and_process_data(airbnb_path, idealista_path)
        
        analyses = []
        # Open text file for writing analyses
        with open(output_text_path, 'w', encoding='utf-8') as text_file:
            for i, (_, airbnb_row) in enumerate(airbnb_df.iterrows(), 1):
                print(f"\nAnalyzing Property {i}/{len(airbnb_df)}:")
                analysis = self.generate_market_analysis(airbnb_row, idealista_df)
                analyses.append(analysis)
                
                # Write analysis to text file with property identifier
                text_file.write(f"\n{'='*80}\n")
                text_file.write(f"Property ID: {airbnb_row['Property ID']}\n")
                text_file.write(f"Neighborhood: {airbnb_row['Neighborhood']}\n")
                text_file.write(f"{'='*80}\n\n")
                text_file.write(analysis)
                text_file.write("\n\n")
                
                print(f"Analysis completed for Property {i}")
        
        # Add analyses to Airbnb DataFrame
        airbnb_df['Market_Analysis'] = analyses
        
        # Save CSV results
        print(f"Saving analysis to {output_csv_path}...")
        airbnb_df.to_csv(output_csv_path, index=False)
        print("Analysis saved successfully!")
        print(f"Text analyses saved to {output_text_path}")


def main():
    try:
        analyzer = PropertyMarketAnalyzer(
            model_name="meta-llama/Llama-2-7b-chat-hf"
        )
        
        # Update paths to your CSV files
        airbnb_path = 'airbnb_turin_properties.csv'
        idealista_path = 'Idealista_sample.csv'
        output_csv_path = 'market_analysis_results.csv'
        output_text_path = 'market_analyses.txt'
        
        analyzer.analyze_market(airbnb_path, idealista_path, output_csv_path, output_text_path)
        
    except Exception as e:
        print(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
