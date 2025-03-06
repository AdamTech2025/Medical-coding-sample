from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os
from dotenv import load_dotenv
from .llms.model import MedicalCodingExtractor, GROQ_API_KEY

load_dotenv()

def index(request):
    context = {
        'result': None,
        'error': None
    }
    
    if request.method == 'POST':
        try:
            clinical_text = request.POST.get('clinical_text')
            
            if not clinical_text:
                context['error'] = 'No clinical text provided'
            else:
                # Initialize the extractor with Groq
                extractor = MedicalCodingExtractor(api_key=GROQ_API_KEY)
                
                # Process the document
                result = extractor.process_ehr_document(clinical_text)
                
                if result.get('status') == 'success':
                    context['result'] = result['data']
                    print(context['result'])
                else:
                    context['error'] = result.get('error', 'Failed to process clinical data')
                
        except ValueError as e:
            context['error'] = f"Configuration error: {str(e)}"
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'core/index.html', context)

@csrf_exempt
def process_medical_text(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        clinical_text = data.get('clinical_text')
        
        if not clinical_text:
            return JsonResponse({'error': 'No clinical text provided'}, status=400)
            
        # Initialize the extractor with Groq
        extractor = MedicalCodingExtractor(api_key=GROQ_API_KEY)
        
        # Process the document
        result = extractor.process_ehr_document(clinical_text)
        
        return JsonResponse(result)
        
    except ValueError as e:
        return JsonResponse({'error': f"Configuration error: {str(e)}"}, status=500)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

