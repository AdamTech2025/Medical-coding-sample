import openai
import json
from datetime import datetime
from django.conf import settings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

class MedicalCodingExtractor:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("API key must be provided")
        self.client = openai.OpenAI(api_key=api_key)
        
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
        """Extract medical codes directly from text using GPT-4"""
        try:
            prompt = f"""As a medical coding expert, analyze this clinical text and extract all relevant ICD-10-CM and CDT codes. Provide accurate codes with verified descriptions.

Clinical text: "{text}"

Please provide your analysis in this EXACT format:

ICD-10-CM (Diagnosis):
[CODE] – [DESCRIPTION]
(List each diagnosis code on a new line)

CDT/CPT (Procedure):
[CODE] – [DESCRIPTION]
(List each procedure code on a new line)

Explanation:
1. Diagnosis Codes:
- [CODE]: Detailed explanation of why this diagnosis code was selected
(Provide explanation for each diagnosis code)

2. Procedure Codes:
- [CODE]: Detailed explanation of why this procedure code was selected, including any relevant measurements, anatomical considerations, or other factors
(Provide explanation for each procedure code)

Note: Ensure all codes are current and verified. Include only codes that are explicitly supported by the clinical documentation."""
            
            completion = self.client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {"role": "system", "content": """You are an expert medical coder specializing in ICD-10-CM and CDT/CPT coding. Your task is to:

1. First identify all documented diagnoses and procedures
2. Select the most specific and accurate codes
3. Format the response exactly as requested with clear separation between diagnoses and procedures
4. Provide detailed explanations for each code selection
5. Only include codes that are explicitly supported by the documentation
6. Use official code descriptions
7. Verify all codes are current and active

Common Dental/Medical Code References:
- D2391: Resin-based composite - one surface, posterior
- D2392: Resin-based composite - two surfaces, posterior
- D2393: Resin-based composite - three surfaces, posterior
- D2394: Resin-based composite - four or more surfaces, posterior
- K02.9: Dental caries, unspecified
- K04.0: Pulpitis
- K04.7: Periapical abscess without sinus

Always ensure codes match the documented conditions and procedures exactly."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent, accurate responses
                max_tokens=2000   # Increased token limit for comprehensive responses
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
