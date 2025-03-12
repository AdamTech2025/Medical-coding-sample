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
            prompt = f"""As a medical coding expert, analyze this clinical text and extract all relevant CPT codes with high precision. Focus on accurate procedure code selection based on the exact procedures described.

Clinical text: "{text}"

IMPORTANT CPT CODE SELECTION GUIDELINES:
- For excision of benign tumor or cyst of mandible by enucleation and/or curettage, use 21040
- For application of interdental fixation device, use 21110
- Use the most specific code that fully describes the procedure performed
- Consider anatomical site, approach, technique, and complexity
- Refer to official CPT code descriptions for accurate selection

Please provide your analysis in this format:

CPT CODES:
[CODE] – [EXACT OFFICIAL DESCRIPTION] (Source: AMA CPT®)
- Detailed justification for why this specific code matches the procedure described
- Evidence from the clinical text that supports this code selection
- Any relevant anatomical considerations or technical aspects

RATIONALE:
- Clear explanation connecting the clinical scenario to the selected CPT code(s)
- Identification of key procedure components that determined code selection
- Explanation of why alternative codes were not selected (if relevant)

Note: Ensure complete accuracy in code selection. When in doubt between similar codes, explain the distinction and why one code is more appropriate than another."""
            
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are an expert medical coder specializing in CPT coding for medical procedures. Your task is to:

1. Carefully analyze clinical scenarios to identify exactly what procedures were performed
2. Select the most specific and accurate CPT code(s) for those procedures
3. Provide the exact official CPT code description from the AMA
4. Justify your code selection with clear references to the clinical text
5. Prioritize accuracy over comprehensiveness - it's better to provide fewer, highly accurate codes than many questionable ones
6. Pay special attention to anatomical details, approach method, and procedure complexity
7. When similar codes exist (e.g., 21040 vs 21110), clearly explain why one is more appropriate

COMMON CPT CODE REFERENCE:
- 21040: Excision of benign tumor or cyst of mandible, by enucleation and/or curettage
- 21110: Application of interdental fixation device for conditions other than fracture or dislocation
- 31231: Nasal endoscopy, diagnostic, unilateral or bilateral
- 99213: Office or other outpatient visit, established patient, low to moderate complexity

Always start by identifying the exact procedure(s) performed and match to the most specific code."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent, accurate responses
                max_tokens=2000   # Significantly increased token limit for comprehensive responses
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
