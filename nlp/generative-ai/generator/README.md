# <center> DATA GENERATOR
The Chat Generator Notebook is a tool designed to **generate synthetic conversational datasets using a LLM**. It supports custom project configurations, dynamic prompt generation, and conversation simulation while leveraging project-specific knowledge bases and templates.
    
The notebook allows for scalable conversation generation and labeling, suitable for **training or testing ML models**.

It includes the following features:
- **Dynamic Prompting**: Supports system and user prompt templates with customizable injections.
- **Knowledge Base Integration**: Applies chunking strategies to project-specific knowledge files for contextual conversation generation.
- **Inference Pipeline**: Interacts with an external LLM API to generate conversational exchanges with adjustable parameters (e.g., temperature, top-p).
- **Synthetic Data Generation**: Generates labeled conversation datasets using customizable templates. Users can configure key aspects of the conversation format, including:
    - The maximum number of exchanges per conversation
    - Participant names
    - The overall structure and flow of the conversation
    
## Projects
In the `projects` folder, and under a specific 'project_name', you will find the following files and directories:
### config/
- `projects/<project_name>/config/chunk-strategy.json`: maps files in knowledge-base with chunk-strategy paramters
- `projects/<project_name>/config/generation-paramters.json`: stores possible range values for inference parameters like temperature, top_k, top_p, etc.
- `projects/<project_name>/config/output-format.json`: template to structure output file.
### examples/
- `projects/<project_name>/examples/examples.txt`: contains handwritten examples to add to the synthetic data.
### generated/
- `projects/<project_name>/generated/<%y-%m-%d-%Hh%Mm%Ss>.json`: generated dataset.
- `projects/<project_name>/generated/<%y-%m-%d-%Hh%Mm%Ss>.csv`: generated dataset.
### knowledge/
- `projects/<project_name>/knowledge/<topic_name>/<document>`: document with domain knowledge to define in-context conversational topics.
### prompting/
- `projects/<project_name>/prompting/system-prompt.txt`: system-prompt for the LLM that will generate the conversations.
- `projects/<project_name>/prompting/user-prompt.txt`: user-prompt for the LLM that will generate the conversations.
- `projects/<project_name>/prompting/templates.txt`: contains different template styles for the conversations.
- `projects/<project_name>/prompting/injections`: directory to store possible values for variables in user and system prompt, enabling dynamic prompting.
    
## Usage Instructions
1. Prerequisites
   
    Ensure the required dependencies are installed:
    ```
    !pip install -r requirements.txt
    ```
    Set up environment variables for API access:
    - LLM_API_URL: URL of the LLM API endpoint.
    - LLM_API_KEY: API key for authentication.
    - LLM_NAME: Name of the LLM to use.

2. Input Setup

    Run the notebook and input the following values when prompted:
    - Project Name: Name of the project folder under the projects/ directory.
    - Iterations: Number of synthetic conversations to generate.
    - Max Exchanges: Maximum number of exchanges per generated conversation.

3. Prepare Project Data

    Ensure the project folder includes:
    - config/: Contains configuration files like chunk-strategy.json and output-format.json.
    - prompting/: Contains .txt files for prompt templates and injections.
    - knowledge/: Stores files to be chunked for contextual conversation data.
    - examples/: A .txt file with manually created examples for additional dataset diversity.
    
## To-Do Enhancements
- **Token Counting**: Incorporate token limits to prevent overflow during API calls and estimate inference cost.
- **Deduplication**: Add similarity checks to remove near-duplicate examples.
