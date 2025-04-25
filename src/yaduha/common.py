from typing import Optional
import openai
import os
from dotenv import load_dotenv
import pathlib

# Load environment variables from .env file
load_dotenv(pathlib.Path(__file__).parent.parent / '.env')
load_dotenv(pathlib.Path.cwd() / '.env')


def get_openai_client(api_key: Optional[str] = None) -> openai.Client:
    """Get OpenAI client with API key from environment variable.

    Args:
        api_key (str, optional): OpenAI API key. If not provided, it will be loaded from the environment variable.

    Returns:
        openai.Client: OpenAI client instance.

    Raises:
        ValueError: If the API key is not set in the environment variable.
    """
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("API key not provided and OPENAI_API_KEY environment variable not set.")
    return openai.Client(api_key=api_key)
