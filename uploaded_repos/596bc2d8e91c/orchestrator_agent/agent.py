from google.adk.agents import LlmAgent, Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.models.google_llm import Gemini
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from orchestrator_agent.document_parser_agent import parse_medical_document
from orchestrator_agent.fair_price_research_agent import fair_price_research_agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import ToolContext, load_artifacts 
from google.adk.tools import agent_tool
from dotenv import load_dotenv
from orchestrator_agent.insurance_advocate_agent import insurance_advocate_agent
import google.genai.types as types
from pydantic import BaseModel, Field
import os
import asyncio
import sys
import logging

class DenialAnalysisItem(BaseModel):
    denied_item: str = Field(..., description="Name of the denied service or item")
    denied_amount: str | None = Field(None, description="Amount that was denied")
    denial_reason: str = Field(..., description="Reason given for denial")
    policy_terms_found: str = Field(..., description="What the policy actually states")
    verdict: str = Field(..., description="Whether denial is justified or not")

class AppealStrategy(BaseModel):
    is_appeal_recommended: bool = Field(..., description="Whether appeal is recommended")
    appeal_deadline: str | None = Field(None, description="Deadline for appeal")
    required_documents: list[str] = Field(default_factory=list, description="Documents needed for appeal")
    appeal_process: str | None = Field(None, description="Steps for appeal process")
    success_likelihood: str | None = Field(None, description="Likelihood of success")
    key_arguments: list[str] = Field(default_factory=list, description="Key points for appeal")

class InsuranceAnalysisResponse(BaseModel):
    insurance_company: str = Field(..., description="Name of the insurance company")
    policy_name: str = Field(..., description="Policy name")
    policy_number: str = Field(..., description="Policy number")
    denial_analysis: list[DenialAnalysisItem] = Field(..., description="Analysis of each denied item")
    appeal_strategy: AppealStrategy = Field(..., description="Appeal strategy and recommendations")
    next_steps: list[str] = Field(..., description="Actionable next steps")

# Suppress non-critical warnings
logging.basicConfig(level=logging.DEBUG)

# Fix for Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

async def process_user_file(tool_context: ToolContext) -> dict:
    '''
    Checks for and saves uploaded images to the uploads folder. 
    Call this tool FIRST on every user message to detect if files were uploaded.

    Arguments:
        tool_context: The ADK context used to load the artifact.

    Returns:
        Dictionary containing:
        - "has_files": bool - Whether any files were found
        - "message": str - Status message
        - "folder_path": str - Where files were saved (if any)
        - "file_count": int - Number of files saved
    '''
    d = {'has_files': False, 'message': '', 'folder_path': '', 'file_count': 0}

    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("orchestrator_agent/uploads", exist_ok=True)
        
        # Get list of artifacts - MUST USE AWAIT
        artifacts = await tool_context.list_artifacts()
        
        if not artifacts or len(artifacts) == 0:
            d['message'] = "No files were uploaded with this message."
            return d
        print(f"Found {len(artifacts)} uploaded file(s). Saving...")
        saved_files = []
        for artifact in artifacts:
            try:
                # Load artifact - MUST USE AWAIT
                artifact_content = await tool_context.load_artifact(filename=artifact)
                file_name = artifact_content.inline_data.display_name 
                data_bytes = artifact_content.inline_data.data
                
                # Save to disk
                file_path = os.path.join("orchestrator_agent/uploads", file_name)
                with open(file_path, 'wb') as f:
                    f.write(data_bytes)
                saved_files.append(file_name)
            except Exception as e:
                d['message'] = f"Error loading artifact {artifact}: {str(e)}"
                return d

        d['has_files'] = True
        d['file_count'] = len(saved_files)
        d['message'] = f"Successfully saved {len(saved_files)} file(s): {', '.join(saved_files)}"
        d['folder_path'] = "orchestrator_agent/uploads"
        tool_context
        return d

    except Exception as e:
        d['message'] = f"Error processing files: {str(e)}"
        return d



model = "gemini-2.5-flash-lite"
APP_NAME = "medical_advocate_orchestrator_agent"

root_agent = LlmAgent(
    name="medical_advocate_orchestrator_agent",
    model=Gemini(model=model), 
    description="Orchestrator Agent to help answer questions & co-ordinate with patients",
    instruction="""
You are a medical billing advocate orchestrator.

WORKFLOW:

Step 1: Call process_user_file()
Step 2: Call parse_medical_document()
Step 3: Based on the output, call the appropriate agents:
    - If only medical bill: call fair_price_research_agent
    - If only denial letter: call insurance_advocate_agent
    - If both: call fair_price_research_agent AND insurance_advocate_agent one after the other

Step 4: After all agent tools have run, you MUST:
    - Collect the responses from each agent
    - Present a final, clear, organized summary to the user
    - Include both the price analysis and the denial/appeal analysis
    - Use headings, bullet points, and formatting for clarity

**Example Final Output:**

---
**Medical Bill Price Analysis**
[Paste the output from fair_price_research_agent here]

---
**Insurance Denial & Appeal Analysis**
[Paste the output from insurance_advocate_agent here]

**Medical Bill and Insurance denial uploaded together**
[Paste the summary of both fair_price_research_agent and insurance_advocate_agent here]

---

Always present BOTH analyses if both documents were uploaded. Do not skip or merge them. Your job is to coordinate and summarize the results for the user.
""",
    tools=[
        process_user_file, 
        parse_medical_document,
        agent_tool.AgentTool(agent=fair_price_research_agent),
        agent_tool.AgentTool(agent=insurance_advocate_agent)
    ]
)


# Set up Artifact Service for handling   uploaded files
artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    session_service=session_service,
    artifact_service=artifact_service,
    app_name="medical_advocate_orchestrator_agent",
    plugins=[SaveFilesAsArtifactsPlugin()],
)

async def main():
    # When user uploads a file through ADK web, it comes as a Part with inline_data
    # Example of how an uploaded image would be represented:
    
    response = await runner.run_debug(
        "I've uploaded a medical bill image. Can you help me understand if I'm being overcharged?"
    )


if __name__ == "__main__":
    asyncio.run(main())