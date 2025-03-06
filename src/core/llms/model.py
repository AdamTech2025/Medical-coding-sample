import groq
import json
from datetime import datetime
from django.conf import settings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_DEEPSEEK_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable must be set")

class MedicalCodingExtractor:
    def __init__(self, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError("API key must be provided")
        try:
            # Try to initialize without any extra parameters
            self.groq_client = groq.Groq(api_key=api_key)
        except TypeError as e:
            # If there's a TypeError about unexpected arguments, try with base_url only
            if 'proxies' in str(e):
                print("Warning: Proxy settings detected but not supported. Initializing without proxies.")
                self.groq_client = groq.Groq(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            else:
                raise
        
    def process_ehr_document(self, ehr_text: str) -> dict:
        """
        Process raw EHR text and extract medical codes
        
        Args:
            ehr_text (str): Raw EHR text
            
        Returns:
            dict: Processed results with codes and metadata
        """
        try:
            # Extract codes directly from text
            result = self._extract_codes_from_text(ehr_text)
            
            return {
                "status": "success",
                "data": result,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "data": None,
                "error": str(e)
            }
    
    def _extract_codes_from_text(self, text: str) -> str:
        """Extract medical codes directly from text using Groq"""
        try:
            prompt = f"""As a medical coding expert, analyze this clinical text and extract all relevant ICD-10 and CPT codes.

Clinical text: "{text}"

Provide a detailed analysis with proper formatting and alignment. Follow this exact format:

=== Medical Conditions Found ===
1. [Condition Name] ([ICD-10 Code] - [Description])
   • Detailed explanation or notes about the condition
   • Additional relevant information if applicable

2. [Condition Name] ([ICD-10 Code] - [Description])
   • Detailed explanation or notes about the condition
   • Additional relevant information if applicable

=== Medical Procedures Performed ===
1. [Procedure Name] ([CPT Code] - [Description])
   • Purpose and details of the procedure
   • Additional relevant information if applicable

2. [Procedure Name] ([CPT Code] - [Description])
   • Purpose and details of the procedure
   • Additional relevant information if applicable

=== Additional Notes ===
• Any relevant updates to codes or special considerations
• Any other important clinical context

Example output:
=== Medical Conditions Found ===
1. Severe Persistent Asthma (J45.50 - Severe persistent asthma)
   • Characterized by continuous symptoms
   • Significant airflow limitation present

2. Type 2 Diabetes (E11.9 - Type 2 diabetes mellitus without complications)
   • No specified complications noted
   • Routine monitoring recommended

=== Medical Procedures Performed ===
1. Spirometry (94010 - Spirometry complete)
   • Comprehensive pulmonary function testing
   • Used to assess severity of airflow obstruction

2. HbA1c Test (82950 - Glucose test, post-glucose dose)
   • Glycemic control assessment
   • Standard diabetes monitoring procedure

=== Additional Notes ===
• All codes verified against current year's updates
• Follow-up recommended in 3 months"""
            
            completion = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a medical coding expert specializing in ICD-10 and CPT code extraction. Format your responses with precise alignment and clear section breaks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response = completion.choices[0].message.content.strip()
            
            # If response is empty, raise an error
            if not response:
                raise ValueError("No analysis generated")
                
            return response
                
        except Exception as e:
            print(f"Error extracting codes: {str(e)}")
            return """=== Error in Analysis ===
Unable to process the clinical text. Please check the input and try again.

Common Issues:
• Missing or unclear clinical information
• Invalid text format
• System processing error

Please ensure the input text contains clear medical conditions and procedures."""
