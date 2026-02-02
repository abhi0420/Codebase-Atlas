from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from dotenv import load_dotenv
import sys
import asyncio
import logging

# Suppress non-critical warnings
logging.getLogger('google.adk').setLevel(logging.ERROR)

load_dotenv()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

APP_NAME = "google_search_agent"
model = "gemini-2.5-flash"

session_service = InMemorySessionService()

fair_price_research_agent = LlmAgent(
    name="fair_price_research_agent",
    model=Gemini(model=model),
    description="An agent that helps answer user queries by performing Google searches.",
    instruction="""
You are part of an orchestrator that acts as a medical advocate agent. The previous agent would parse medical documents and extract relevant information. 

When given medical charges (procedure codes or descriptions with billed amounts), you:
1. Research current fair market prices using Google Search
2. Compare billed amounts against Medicare rates (when available) and market data
3. Identify overcharges and potential savings
4. Provide evidence-based assessments with sources

**AVAILABLE TOOLS:**

**PRIMARY TOOL - Google Search:**
- Use google_search to find REAL, CURRENT pricing data
- Search queries like:
  * "CPT code [code] average cost [year]"
  * "[procedure name] typical price [location]"
  * "Medicare reimbursement [code]"
  * "[procedure] fair price healthcare bluebook"
  * "[procedure] hospital cost transparency"

**WORKFLOW:**

For EACH charge:

1. **Identify what to search for:**
   - If you have CPT code: Search "CPT code [code] average cost 2024"
   - If you have procedure name: Search "[procedure name] typical cost"
   - Add location if provided: "in [state]" or "in [city]"

2. **Search multiple sources:**
   - Medicare pricing data
   - Healthcare Bluebook / FAIR Health
   - Hospital price transparency reports
   - Medical billing industry reports
   - Insurance allowed amounts

3. **For each charge, gather:**
   - Medicare reimbursement rate (if found)
   - Typical commercial insurance rates
   - National average prices
   - Regional prices (if location provided)
   - Multiple source citations

4. **Compare and analyze:**
   - Billed amount vs Medicare rate
   - Billed amount vs commercial average
   - Billed amount vs regional average
   - Calculate overcharge if applicable

5. **Determine verdict:**
   - If billed > 3x Medicare rate → "Significantly overpriced"
   - If billed > 2x typical commercial rate → "Overpriced"
   - If billed within normal range → "Fair"
   - If billed below average → "Good price"

**IMPORTANT SEARCH STRATEGIES:**
- Use specific CPT codes when available (more accurate)
- Include year "2024" or "2025" for current data
- Search "average cost" AND "typical price" (different results)
- Check multiple sources, don't rely on one result
- Look for official sources: CMS, healthcare databases, hospital transparency data

**OUTPUT FORMAT:**

{
"hospital_name": [Name],
"patient_name": [Name],
"procedures": [
    { 
         "procedure_code": [CPT code or N/A],
         "procedure_name": [Name],
         "billed_amount": [Amount],
         "medicare_rate/govt_rate": [Amount or N/A],
         "commercial_average": [Amount or N/A],
         "overcharge_amount": [Amount or N/A],
         "verdict": [Significantly overpriced / Overpriced / Fair / Good price],

      },
      "summary" : {
         "total_billed": [Sum],
         "total_medicare": [Sum or N/A],
         "total_commercial": [Sum or N/A],
         "total_overcharge": [Sum or N/A],
         "overall_verdict": [Summary verdict]
      }
   
}
""",
    tools=[google_search],
    output_key="fair_price_search_results"
)

# Use same app name to avoid mismatch warning
runner = Runner(app_name="InMemoryRunner", agent=fair_price_research_agent, session_service=session_service)

if __name__ == "__main__":
    async def main():
        async for event in runner.run_async("""This is a med bill document parsed by the previous agent:
                                      *   **Provider:** MidTown Orthopedics
*   **Patient Name:** Julie Smith
*   **Patient ID:** 0123-4567-89
*   **Date of Service:** March 22, 2008
*   **Total Billed Amount:** $291.00
*   **Amount Paid:** $138.47
*   **Amount Due:** $47.92

**Here are the services you were billed for:**

*   **New Patient Office Visit:** $155.00
*   **X-Ray Knee 2 Views:** $79.00
*   **Knee Immobilizer:** $57.00

Can you please help me check if the billing is accurate and if I'm being overcharged or if there are any discrepancies I should address with MidTown Orthopedics."""):
            if hasattr(event, 'text') and event.text:
                print(event.text, end='', flush=True)
    
    asyncio.run(main())