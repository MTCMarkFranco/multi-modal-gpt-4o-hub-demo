# Multi-Modal GPT-4o Hub Demo

This project demonstrates the use of Azure OpenAI's GPT-4o model to create and analyze images using multi-modal capabilities. Specifically, it generates a stop sign image programmatically, encodes it in base64, and then uses GPT-4o to analyze the image for correctness based on predefined visual attributes.

## Project Goals

- **Generate Images**: Programmatically create a stop sign image using OpenCV.
- **Analyze Images**: Use Azure OpenAI's GPT-4o multi-modal capabilities to analyze the generated image.
- **Agent Collaboration**: Demonstrate interaction between multiple autonomous agents (`StopSignCreator`, `StopSignReviewer`, and `UserProxyAgent`) using the Autogen framework.

## Key Components

- **Image Generation** (`create_and_inject_stop_sign`):
    ```python
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    # Generate octagonal stop sign shape
    points = np.array([...], np.int32)
    cv2.fillPoly(img, [points], (0, 0, 255))
    cv2.putText(img, 'STOP', (120, 220), font, 2, (255, 255, 255), 6)
    ```

- **Image Analysis** (`describe_image_with_llm`):
    ```python
    response = await az_model_client.create(messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ])
    ```

- **Agent Setup**:
    ```python
    image_agent = MultimodalConversableAgent(...)
    reviewer_agent = MultimodalConversableAgent(...)
    user_proxy = UserProxyAgent(...)
    ```

## Setup Instructions

### Environment Variables

Create a `.env` file in the root of your project directory with the following variables (do **not** include sensitive values here):

```properties
COMPLETIONS_MODEL=
DEPLOYMENT_NAME=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
OPENAI_API_VERSION=
```

### Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

### Running the Project

Run the main script:

```bash
python main.py
```

Ensure your Azure OpenAI resources are properly configured and accessible.

## Azure Best Practices

- Store sensitive keys and endpoints securely using environment variables or Azure Key Vault.
- Regularly rotate your Azure OpenAI API keys.
- Follow Azure's recommended security guidelines for managing resources and access control.
