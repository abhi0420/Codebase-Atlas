from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.models.google_llm import Gemini
from dotenv import load_dotenv
import sys


load_dotenv()

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app_name = "insurance_advocate_agent"
model = "gemini-2.5-flash"

insurance_advocate_agent = LlmAgent(
    name="insurance_advocate_agent",
    model=Gemini(model=model),
    description="An agent that acts as a medical insurance advocate to help patients understand and challenge their medical bills and insurance claims.",
    instruction="""
You are part of an orchestrator that acts as a medical advocate agent. 

When given an insurance denial or EOB, you:
1. Research the specific insurance company's policy terms using Google Search
2. Compare denial reasons against actual policy coverage
3. Determine if denial is justified or appealable
4. Provide evidence-based appeal strategy with sources

**AVAILABLE TOOLS:**

**PRIMARY TOOL - Google Search:**
- Use google_search to find REAL, CURRENT policy information
- Search queries like:
  * "[Insurance Company] [Policy Name] coverage terms PDF"
  * "[Insurance Company] [Policy Name] room rent limit 2024"
  * "[Insurance Company] [Policy Name] claim documentation requirements"
  * "[Insurance Company] appeal process"
  * "[Insurance Company] [specific denial reason] coverage policy"

**WORKFLOW:**

For EACH denial reason:

1. **Identify what to search for:**
   - Extract insurance company name (e.g., "HDFC ERGO")
   - Extract policy name/type (e.g., "Health Suraksha Silver Plan")
   - Extract denial reason (e.g., "room rent exceeds limit")

2. **Search for policy terms:**
   - Search "[Insurance Company] [Policy] [specific coverage area]"
   - Examples:
     * "HDFC ERGO Health Suraksha Silver Plan room rent limit"
     * "Star Health Comprehensive Plan claim requirements"
     * "ICICI Lombard claim documentation needed"

3. **Search for each denied item:**
   - If denied for room rent → search policy's room rent limits
   - If denied for missing docs → search policy's claim requirements
   - If denied for procedure → search policy's coverage for that procedure
   - If denied for threshold → search policy's sub-limits

4. **Search for appeal process:**
   - "[Insurance Company] claim appeal process"
   - "[Insurance Company] grievance redressal"

5. **Compare and analyze:**
   - What does the policy ACTUALLY say?
   - Does the denial align with policy terms?
   - Are there loopholes or exceptions in policy?

6. **Determine verdict:**
   - If policy supports denial → "Justified - denial aligns with policy terms"
   - If policy contradicts denial → "Unjustified - policy covers this"
   - If unclear → "Needs clarification - appeal recommended"

**IMPORTANT SEARCH STRATEGIES:**
- Always include insurance company name AND policy name
- Look for official policy documents (PDFs)
- Check insurance company's website
- Search IRDA (Insurance Regulatory) guidelines
- Look for consumer forums with similar cases
- Include year for current policy terms

**OUTPUT FORMAT:**

{
  "insurance_company": [Name],
  "policy_name": [Policy],
  "policy_number": [Number if available],
  "denial_analysis": [
    {
      "denied_item": [Service/Item name],
      "denied_amount": [Amount],
      "denial_reason": [Reason given],
      "policy_terms_found": [What you found from search],
      "verdict": [Justified / Unjustified / Needs clarification]
    }
  ],
  "appeal_strategy": {
    "is_appeal_recommended": [true/false],
    "appeal_deadline": [Date if found],
    "required_documents": [List from search],
    "appeal_process": [Steps from search],
    "success_likelihood": [High/Medium/Low],
    "key_arguments": [Points to make in appeal]
  },
  "next_steps": [Actionable list]
}

**CRITICAL:**
- You MUST call google_search multiple times (at least 3-5 searches)
- Search for EACH denied item separately
- Don't make assumptions - find actual policy documents
- Provide source links for everything you claim
""",
    tools=[google_search]
)

runner = Runner(
    app_name=app_name, 
    agent=insurance_advocate_agent, 
    session_service=InMemorySessionService()
)

async def main():
    response = await runner.run_debug("""
HDFC ERGO General Insurance Company Ltd.
Claims Review Department
1st Floor, HDFC House, Lower Parel, Mumbai – 400013

Date: 27-Nov-2025

To:
Mr. Ramesh Kumar
Patient ID: RK-45892
                                      


Subject: Claim Rejection Notification
Policy Number: 2314/98374652/00
Policy Name: HDFC ERGO Health Suraksha Silver Plan

Dear Mr. Kumar,

We have completed the review of your claim submitted on 22-Nov-2025 for reimbursement of medical expenses incurred at Sunrise Care & Diagnostics Pvt. Ltd.

After evaluating the documents provided, the following items have been rejected:

Rejected Claim Line Items

Room Rent Charges – ₹5,500.00
Reason: The billed amount exceeds the admissible limit under the policy.

Blood Test Panel – ₹3,200.00
Reason: Supporting laboratory reports required for verification were not provided.

X-Ray (Chest PA + Lateral) – ₹4,800.00
Reason: The claimed amount exceeds the allowable charge threshold.

Medicines & Consumables – ₹6,500.00
Reason: Itemized breakdown with drug names, quantities, and batch numbers was not submitted.

Consultation & Procedure Fee – ₹7,000.00
Reason: Absence of doctor’s notes / treatment summary required for claim validation.

Final Status: CLAIM REJECTED

You may resubmit the claim with the required supporting documents within 30 days for reconsideration.

For assistance, please contact our Claims Support team at the number listed on your policy document.

Sincerely,
Claims Review Officer
HDFC ERGO General Insurance Company Ltd
""")



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())