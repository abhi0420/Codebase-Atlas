from google import genai
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, Union
from dotenv import load_dotenv
from google.genai import types
import json 
from google.adk.artifacts import InMemoryArtifactService
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

model = "gemini-2.5-flash"

class Charge(BaseModel):
    code : str = Field(description="The medical service code",default="99215")
    description : str = Field(description="Description of the medical service",default="Office visit - Level 5")
    amount : float = Field(description="Amount billed for this service")
    diagnosis_code : str = Field(description="Associated diagnosis code")

class MedicalBill(BaseModel):
    document_type: str = Field(default="medical_bill")
    hospital_name : str = Field(description="Name of the medical provider",default="General")
    city : str = Field(description="Location of the medical provider")
    country : str = Field(description="Country of the medical provider")
    patient_name: str = Field(description="Name of the patient")
    patient_id : str = Field(description="Patient Identifier",default="000")
    date_of_service : str = Field(description="Date of service")
    charges : list[Charge] = Field(description="List of charges in the bill")   
    total_billed : float = Field(description="Total amount billed")
    amount_paid : float = Field(description="Amount already paid")
    amount_due : float = Field(description="Amount due")
    due_date : str = Field(description="Due date for the payment")

class EOCoverage(BaseModel):
    service : str = Field(description="Description of the service")
    billed_amount : float = Field(description="Amount billed for the service")
    allowed_amount : float = Field(description="Amount allowed by insurance")
    paid_by_insurance : float = Field(description="Amount paid by insurance")
    patient_responsibility : float = Field(description="Amount patient has to pay")

class InsuranceEOB(BaseModel):
    doc_type : str = Field(default="insurance_eob")
    insurance_company : str = Field(description="Name of the insurance provider")
    policy_number : str = Field(description="Insurance policy number")
    patient_name : str = Field(description="Name of the patient")
    claim_number : str = Field(description="Claim number")
    date_of_service : str = Field(description="Date of service")
    coverage_details : list[EOCoverage] = Field(description="Details of coverage for each service")
    total_patient_responsibility : float = Field(description="Total amount patient has to pay")

class DenialLetter(BaseModel):
    doc_type : str = Field(default="denial_letter")
    insurance_company : str = Field(description="Name of the insurance provider")
    patient_name : str = Field(description="Name of the patient")
    claim_number : str = Field(description="Claim number")
    policy_number : str = Field(description="Insurance policy number")
    date_of_service : str = Field(description="Date of service")
    denied_services : list[str] = Field(description="List of services that were denied")
    denial_reasons : list[str] = Field(description="Reasons for denial provided by insurance")
    denial_date : str = Field(description="Date when the denial was issued")
    appeal_deadline : str = Field(description="Deadline to file an appeal")
    appeal_instructions : str = Field(description="Instructions on how to file an appeal")

# Create the root agent - this is the entry point
def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from model response"""
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    # Parse JSON
    return json.loads(text)



def _get_mime_type(file_path: str) -> str:
        """Utility to get MIME type based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        elif ext == '.pdf':
            return 'application/pdf'
        else:
            return 'application/octet-stream'
        
def _read_file(file_path: str) -> bytes:
        """Utility to read file content as bytes"""
        with open(file_path, 'rb') as f:
            return f.read()
        

      

def parse_medical_document() -> Dict[str, Any]:
    """
    Parse medical documents (bill, EOB, or denial letter) using Gemini Vision.
    """
    
    image_files = []  # Track files for cleanup
    
    try:
        # Get all image files from uploads folder
        uploads_dir = "orchestrator_agent/uploads"
        image_extensions = ('.jpg', '.jpeg', '.png', '.pdf')
        
        if not os.path.exists(uploads_dir):
            return {
                "error": f"Uploads folder not found: {uploads_dir}",
                "details": "Please create the uploads folder and add your medical documents"
            }
        
        image_files = [
            os.path.join(uploads_dir, f) 
            for f in os.listdir(uploads_dir) 
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            return {
                "error": "No image files found in uploads folder",
                "details": f"Checked folder: {uploads_dir}. Supported formats: {image_extensions}"
            }
        
        print(f"📄 Found {len(image_files)} file(s) to process: {[os.path.basename(f) for f in image_files]}")
        
        # Create prompt with schemas
        prompt = f"""You are analyzing {len(image_files)} medical document image(s).

**TASK:** Extract structured data from EACH distinct document type you find.

**DOCUMENT TYPES:**
1. Medical Bill - hospital/provider billing statement
2. Insurance EOB - Explanation of Benefits from insurance company  
3. Denial Letter - insurance claim denial notice

**INSTRUCTIONS:**

1. Examine all {len(image_files)} images carefully
2. Identify EACH distinct document type present
3. Extract complete data for EACH document using the appropriate schema below

**SCHEMAS:**

Medical Bill Schema:
{json.dumps(MedicalBill.model_json_schema(), indent=2)}

Insurance EOB Schema:
{json.dumps(InsuranceEOB.model_json_schema(), indent=2)}

Denial Letter Schema:
{json.dumps(DenialLetter.model_json_schema(), indent=2)}

**OUTPUT FORMAT:**

If you find MULTIPLE document types (e.g., bill AND denial letter):
Return a JSON array:
[
  {{"document_type": "medical_bill", "hospital_name": "...", ...}},
  {{"document_type": "denial_letter", "insurance_company": "...", ...}}
]

If you find only ONE document type (even if multi-page):
Return a single JSON object:
{{"document_type": "medical_bill", "hospital_name": "...", ...}}

**CRITICAL RULES:**
- Use EXACT field names from schemas
- For dates: use "YYYY-MM-DD" format
- For amounts: use numbers only (no $ symbols)
- Extract ALL charges/services, not just first few
- If field is missing/unclear: use null
- NO markdown formatting
- NO code blocks
- NO explanations
- Return ONLY valid JSON

Start your response with either [ or {{ - nothing else.
"""
        
        # Build parts list with all images
        parts = []
        for file_path in image_files:
            file_bytes = _read_file(file_path)
            mime_type = _get_mime_type(file_path)
            parts.append({"inline_data": {"mime_type": mime_type, "data": file_bytes}})
        
        # Add text prompt at the end
        parts.append({"text": prompt})
        
        print("🤖 Calling Gemini API...")
        
        # Call Gemini Vision with all images
        response = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": parts}]
        )
        
        # Parse response - handle different response formats
        response_text = ""
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            response_text = response.candidates[0].content.parts[0].text
        else:
            return {
                "error": "Could not extract text from response",
                "raw_response": str(response)
            }
        
        print(f"📥 Raw response length: {len(response_text)} chars")
        print(f"📥 First 200 chars: {response_text[:200]}")
        
        # Parse response
        try:
            parsed_data = _parse_json_response(response_text)
        except json.JSONDecodeError as e:
            return {
                "error": "Failed to parse JSON from model response",
                "details": str(e),
                "raw_response": response_text[:500]
            }
        
        # Handle both array and single object responses
        if isinstance(parsed_data, list):
            # Multiple documents
            print(f"✅ Parsed {len(parsed_data)} documents")
            for doc in parsed_data:
                doc_type = doc.get('document_type') or doc.get('doc_type')
                if doc_type:
                    doc['document_type'] = doc_type
            
            result = {
                "documents": parsed_data,
                "count": len(parsed_data),
                "processed_files": [os.path.basename(f) for f in image_files]
            }
        else:
            # Single document
            doc_type = parsed_data.get('document_type') or parsed_data.get('doc_type')
            if not doc_type:
                return {
                    "error": "Could not determine document type",
                    "raw_data": parsed_data
                }
            
            print(f"✅ Parsed single document: {doc_type}")
            parsed_data['document_type'] = doc_type
            parsed_data['processed_files'] = [os.path.basename(f) for f in image_files]
            result = parsed_data
        
        # Delete files after successful parsing
        for file_path in image_files:
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete {file_path}: {e}")
        
        print(result)
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": f"Unexpected error during parsing: {e}",
            "details": str(e)
        }


if __name__ == "__main__":
    print("TESTING DOCUMENT PARSER TOOL")
    
    result = parse_medical_document()
    
    if "error" in result:
        print(f" Error: {result['error']}")
        if "raw_response" in result:
            print(f"Raw response: {result['raw_response'][:500]}...")
    else:
        print(f"✅ Success! Document type: {result.get('document_type')}")
        print(json.dumps(result, indent=2))

