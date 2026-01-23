"""Test script for the OllamaAgent.

Prerequisites:
1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. Pull a model: ollama pull llama3.2:3b (or qwen2.5:3b, mistral:7b, etc.)
3. Ensure Ollama is running: ollama serve

Usage:
    python scripts/test_ollama_agent.py
    python scripts/test_ollama_agent.py --model qwen2.5:3b
"""

import argparse
from typing import ClassVar, List, Tuple, Dict
from pydantic import BaseModel, Field
from yaduha.agent.ollama import OllamaAgent
from yaduha.tool import Tool


class Person(BaseModel):
    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


class GetWeather(Tool):
    """Simple tool to test tool calling."""

    name: ClassVar[str] = "get_weather"
    description: ClassVar[str] = "Get the current weather for a city."

    def _run(self, city: str) -> str:
        # Mock weather data
        weather_data = {
            "new york": "Sunny, 72°F",
            "london": "Cloudy, 58°F",
            "tokyo": "Rainy, 65°F",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")

    def get_examples(self) -> List[Tuple[Dict[str, str], str]]:
        return [({"city": "New York"}, "Sunny, 72°F")]


def test_text_response(agent: OllamaAgent) -> None:
    """Test simple text generation."""
    print("\n" + "=" * 50)
    print("Test 1: Simple Text Response")
    print("=" * 50)

    response = agent.get_response(
        messages=[{"role": "user", "content": "Say hello in exactly one sentence."}]
    )

    print(f"Response: {response.content}")
    print(f"Response Time: {response.response_time:.2f}s")
    print(f"Prompt Tokens: {response.prompt_tokens}")
    print(f"Completion Tokens: {response.completion_tokens}")


def test_structured_output(agent: OllamaAgent) -> None:
    """Test structured output with Pydantic model."""
    print("\n" + "=" * 50)
    print("Test 2: Structured Output (Person model)")
    print("=" * 50)

    response = agent.get_response(
        messages=[
            {
                "role": "user",
                "content": "Generate a random person with a creative fantasy name and age between 20 and 80.",
            }
        ],
        response_format=Person,
    )

    print(f"Response: {response.content}")
    print(f"  Name: {response.content.name}")
    print(f"  Age: {response.content.age}")
    print(f"Response Time: {response.response_time:.2f}s")
    print(f"Prompt Tokens: {response.prompt_tokens}")
    print(f"Completion Tokens: {response.completion_tokens}")


def test_tool_calling(agent: OllamaAgent) -> None:
    """Test tool calling functionality."""
    print("\n" + "=" * 50)
    print("Test 3: Tool Calling (GetWeather)")
    print("=" * 50)

    response = agent.get_response(
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Tokyo?",
            }
        ],
        tools=[GetWeather()],
    )

    print(f"Response: {response.content}")
    print(f"Response Time: {response.response_time:.2f}s")
    print(f"Prompt Tokens: {response.prompt_tokens}")
    print(f"Completion Tokens: {response.completion_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Test OllamaAgent")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--skip-tools",
        action="store_true",
        help="Skip tool calling test (some models don't support it)",
    )
    args = parser.parse_args()

    print(f"Testing OllamaAgent with model: {args.model}")
    print(f"Base URL: {args.base_url}")

    agent = OllamaAgent(
        model=args.model,
        base_url=args.base_url,
    )

    try:
        test_text_response(agent)
        test_structured_output(agent)

        if not args.skip_tools:
            test_tool_calling(agent)
        else:
            print("\n[Skipping tool calling test]")

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Ollama running? Try: ollama serve")
        print(f"  2. Is the model pulled? Try: ollama pull {args.model}")
        print(f"  3. Is the server accessible at {args.base_url}?")
        raise


if __name__ == "__main__":
    main()
