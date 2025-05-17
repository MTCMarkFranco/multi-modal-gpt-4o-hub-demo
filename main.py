import autogen
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
import base64
import cv2
import numpy as np
import io
from PIL import Image
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Load environment variables
load_dotenv()

azure_config = {
    "api_key": os.getenv('AZURE_OPENAI_KEY'),
    "api_version": os.getenv('OPENAI_API_VERSION'),
    "azure_endpoint": os.getenv('AZURE_OPENAI_ENDPOINT'),
    "deployment_name": os.getenv('DEPLOYMENT_NAME'),
    "completions_model": os.getenv('COMPLETIONS_MODEL')
}

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

    az_model_client = AzureOpenAIChatCompletionClient(
                        azure_deployment=azure_config["deployment_name"],
                        model=azure_config["completions_model"],
                        api_version=azure_config["api_version"],
                        azure_endpoint=azure_config["azure_endpoint"],
                        api_key=azure_config["api_key"],
                        temperature=0.4
    )

    response = await az_model_client.create(messages=messages)
    return response.choices[0].message.content


# Create the image creator agent
image_agent = MultimodalConversableAgent(
    name="StopSignCreator",
    system_message="""
            I create stop sign images using python script. and send them to the reviewer agent, 
            who will analyze them by creating a multi modal prompt and getting 5the textual 
            repesentation of the image.
            
            Suggestions for initial image creation lbraries to try:
            1. numpy
            2. PIL
            
            """,
    llm_config={
        "config_list": [{
            "model": azure_config["deployment_name"],
            "api_type": "azure",
            "api_key": azure_config["api_key"],
            "api_version": azure_config["api_version"]
        }]
    }
)

# Create the reviewer agent
reviewer_agent = MultimodalConversableAgent(
    name="StopSignReviewer",
    system_message="""You are a stop sign review specialist to analyze the textual attributes of a stop sign.
    
    GOALS:
    1. Analyze provided stop sign images for the following required attributes:
       - Shape: Must be a regular octagon
       - Color: Must be red with white border
       - Text: Must contain 'STOP' in white letters
       - Text Position: Must be centered
       - Visibility: Must be clear and legible
       - Proportions: Must follow standard traffic sign specifications
    
    2. For each analysis:
       - Provide a detailed breakdown of present attributes
       - List any missing or incorrect attributes
       - If all attributes are correct, approve and recommend terminating the chat
       - If any attributes are missing/incorrect, provide specific code optimization suggestions
    
    3. Format your response as:
       ANALYSIS:
       - Present attributes: [list]
       - Missing/incorrect attributes: [list]
       - Verdict: [APPROVE/NEEDS OPTIMIZATION]
       - Suggestions: [if needed]
       - Recommendation: [TERMINATE/CONTINUE] chat
    
    Be precise and thorough in your visual analysis.""",
    llm_config={
        "config_list": [{
            "model": azure_config["deployment_name"],
            "api_type": "azure",
            "api_key": azure_config["api_key"],
            "api_version": azure_config["api_version"]
        }]
    }
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "use_docker": False,
        "timeout": 60,
        "last_n_messages": 20
    },
)

# Register the describe function with the reviewer agent
reviewer_agent.register_function(
    function_map={
        "describe_image_with_llm": describe_image_with_llm
    }
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, image_agent, reviewer_agent],
    messages=[],
    max_round=30
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={
        "config_list": [{
            "model": azure_config["completions_model"],
            "api_type": "azure",
            "api_key": azure_config["api_key"],
            "api_version": azure_config["api_version"]
        }]
    }
)

# Initial prompt to start the process
async def run_group_chat():
    initial_prompt = """Please follow this process:
    1. StopSignCreator: Generate a stop sign image using python code and test it, then return its base64 representation
    2. StopSignReviewer: Perform a detailed analysis of the image by using your GPT-4o multi-modal capabilities
    3. Based on the reviewer's analysis:
       - If all attributes are correct, approve and end the chat
       - If improvements is needed, provide suggestionsfor updated code, run and test th code and pass it through thr reviewer agent again
    4. Once the reviewer agent approves the image, terminate the chat
    
    Begin with the stop sign creation."""
    
    user_proxy.initiate_chat(
        manager,
        message=initial_prompt
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_group_chat())
