import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core import Image
from dotenv import load_dotenv
import os
from openai import AsyncAzureOpenAI  # or from azure.ai.openai.aio import OpenAIClient if using azure-ai-openai
from playwright.async_api import async_playwright

# Load environment variables
load_dotenv()

azure_config = {
    "api_key": os.getenv('AZURE_OPENAI_KEY'),
    "api_version": os.getenv('OPENAI_API_VERSION'),
    "azure_endpoint": os.getenv('AZURE_OPENAI_ENDPOINT'),
    "deployment_name": os.getenv('DEPLOYMENT_NAME'),
    "completions_model": os.getenv('COMPLETIONS_MODEL')
}

az_model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        model=os.getenv('COMPLETIONS_MODEL'),
        api_version=os.getenv('OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        temperature=0.4
    )

async def describe_image_with_llm(image_url: str) -> str:
    """
    Sends the image to the underlying LLM (GPT-4o) and gets a detailed description.
    """
    prompt = "Describe this image in detail."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    response = await az_model_client.chat.completions.create(
        model=azure_config["deployment_name"],
        messages=messages,
        temperature=0.4
    )
    return response.choices[0].message.content

# async def display_image_in_browser(base64_image: str):
#     """
#     Uses Playwright to display the base64 image in a browser.
#     """
#     async with async_playwright() as p:
#         browser = await p.chromium.launch()
#         page = await browser.new_page()
#         html_content = f"""
#         <html>
#         <body>
#             <h1>Generated Stop Sign</h1>
#             <img src='data:image/png;base64,{base64_image}' alt='Stop Sign'>
#         </body>
#         </html>
#         """
#         await page.set_content(html_content)
#         await page.wait_for_timeout(10000)  # Display for 10 seconds
#         await browser.close()


#     name="StopSignCreator",
#     system_message="""
#         You are a Python script generator. Your task is to create a Python script that generates a stop sign image.
#         The script should be able to run in a local environment and return the base64 representation of the generated image.
#         You will also send the generated image to a web browser for visualization.
#         """,
#     model_client=az_model_client,
#     tools=[{
#         display_image_in_browser
#     }]


stop_sign_reviewer = AssistantAgent(
    name="StopSignReviewer",
    system_message="""You are a stop sign review specialist to analyze the textual attributes of a stop sign.
        #     GOALS:
        #     1. Analyze provided stop sign images for the following required attributes:
        #        - Shape: Must be a regular octagon
        #        - Color: Must be red with white border
        #        - Text: Must contain 'STOP' in white letters
        #        - Text Position: Must be centered
        #        - Visibility: Must be clear and legible
        #        - Proportions: Must follow standard traffic sign specifications
            
        #     2. For each analysis:
        #        - Provide a detailed breakdown of present attributes
        #        - List any missing or incorrect attributes
        #        - If all attributes are correct, approve and recommend terminating the chat
        #        - If any attributes are missing/incorrect, provide specific code optimization suggestions
            
        #     3. Format your response as:
        #        ANALYSIS:
        #        - Present attributes: [list]
        #        - Missing/incorrect attributes: [list]
        #        - Verdict: [APPROVE/NEEDS OPTIMIZATION]
        #        - Suggestions: [if needed]
        #        - Recommendation: [TERMINATE/CONTINUE] chat
            
        #     Be precise and thorough in your visual analysis.""",
    model_client=az_model_client,
    tools=[
        describe_image_with_llm
    ]
)

websurfer_agent = MultimodalWebSurfer(
    name="WebSurfer",
    description="You are a web surfer agent. Your task is rendering the image in a web browser.",
    model_client=az_model_client,
    headless=False
    
)

stop_sign_creator = MagenticOneCoderAgent(
    name="StopSignCreator",
    system_message="""
        You are a Python script generator. Your task is to create a Python script that generates a stop sign image.
        The script should be able to run in a local environment and return the base64 representation of the generated image.
        You will also send the generated image to a web browser for visualization.
        """,
    model_client=az_model_client,
    tools=[
        websurfer_agent
    ]
)


team = MagenticOneGroupChat([stop_sign_creator,stop_sign_reviewer], model_client=az_model_client)
asyncio.run(Console(team.run_stream(task="""Please follow this process:
#     1. StopSignCreator: Generate a stop sign image using python code, test it, and return its base64 representation. Also, send the image to the browser for visualization.
#     2. StopSignReviewer: Perform a detailed analysis of the image by using your GPT-4o multi-modal capabilities
#     3. Based on the reviewer's analysis:
#        - If all attributes are correct, approve and end the chat
#        - If improvements are needed, provide suggestions for updated code, run and test the code, and pass it through the reviewer agent again
#     4. Once the reviewer agent approves the image, terminate the chat
    
#     Begin with the stop sign creation.""")))
