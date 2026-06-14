from groq import Groq
import os
import json
_client : Groq | None = None # Lazy initialization to avoid loading the model multiple times during development

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def get_client():
    global _client # Use the global variable to store the client instance
    if _client is None: # Check if the client has already been initialized
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        _client =Groq(api_key=api_key) # Initialize the client with the API key
    return _client

def chat(prompt : str, system: str = "You are a helpful research assistant.", model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """_summary_
    Single-turn chat. Returns the assistant's reply as a plain string

    Args:
        prompt (str): user entered input text
        system (str, optional): system message the model will use to guide its responses. Defaults to "You are a helpful research assistant.".
        model (str, optional): model to use. Defaults to "llama-3.1-8b-instant".
        temperature (float, optional): temperature for sampling. Defaults to 0.2.
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to 1024.

    Returns:
        str: The generated response from the language model.
    """
    client = get_client() # Get the initialized client instance
    response = client.chat.completions.create( # Create a chat completion using the client
        model = model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
    return response.choices[0].message.content.strip() # Return the content of the response, stripping any leading/trailing whitespace


def chat_json(
        prompt: str,
        system: str = "You are a helpful research assistant.",
        model: str = "llama-3.1-8b-instant", 
        temperature: float = 0.1, 
        max_tokens: int = 1024
    ) -> dict:

    raw_response_text = chat(prompt, system=system, model=model, temperature=temperature, max_tokens=max_tokens) # Get the chat response as text
    clean = raw_response_text.strip() # Clean up the response text by stripping leading/trailing whitespace
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip().strip("```")

    try:
        return json.loads(clean) # Attempt to parse the cleaned response text as JSON and return it as a dictionary
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response as JSON. Cleaned response: {clean}, here is raw output: {raw_response_text}") from e
