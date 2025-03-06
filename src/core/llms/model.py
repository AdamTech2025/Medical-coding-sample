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
            prompt = f"""As a medical coding expert, analyze this clinical text and extract all relevant ICD-10-CM and CPT codes.

Clinical text: "{text}"

Provide a detailed analysis with proper formatting. Follow this exact format:

CPT Codes:
o [CODE] – [DESCRIPTION]
o [CODE] – [DESCRIPTION]

ICD-10-CM Code:
o [CODE] – [DESCRIPTION]

Explanation:
o [CPT CODE]: Detailed explanation of why this code was selected, including any measurements, anatomical considerations, or other relevant factors.
o [CPT CODE]: Detailed explanation for this code, including any clarifications or corrections if needed.
o Diagnosis ([ICD-10 CODE]): Explanation of the diagnosis code selection with specifics about location, type, etc.

Be precise, technical, and follow coding guidelines exactly. Include measurements, anatomical considerations, and any necessary corrections or clarifications in your explanations."""
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=[
                    {"role": "system", "content": """You are a medical coding expert specializing in ICD-10-CM and CPT code extraction. 
Format your responses exactly as follows:

CPT Codes:
o [CODE] – [DESCRIPTION]
o [CODE] – [DESCRIPTION]

ICD-10-CM Code:
o [CODE] – [DESCRIPTION]

Explanation:
o [CODE]: Detailed explanation of why this code was selected, including any measurements, anatomical considerations, or other relevant factors.
o [CODE]: Detailed explanation for this code, including any clarifications or corrections if needed.
o Diagnosis ([CODE]): Explanation of the diagnosis code selection with specifics about location, type, etc.

Be precise, technical, and follow coding guidelines exactly."""},
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
