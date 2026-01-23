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
import requests
from typing import ClassVar, List, Tuple, Dict, Optional
from pydantic import BaseModel, Field
from yaduha.agent.ollama import OllamaAgent
from yaduha.tool import Tool


def get_installed_models(base_url: str = "http://localhost:11434") -> List[Dict]:
    """Fetch list of installed models from Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama server.")
        print("Make sure Ollama is running: ollama serve")
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def select_model_interactive(models: List[Dict]) -> Optional[str]:
    """Display interactive menu to select a model."""
    if not models:
        print("\nNo models installed. Install one with: ollama pull <model>")
        print("Suggested models:")
        print("  ollama pull llama3.2:3b    # Small, fast")
        print("  ollama pull llama3.1:8b    # Medium")
        print("  ollama pull qwen2.5:7b     # Good at structured output")
        return None

    print("\nInstalled models:")
    print("-" * 60)
    for i, model in enumerate(models, 1):
        name = model.get("name", "unknown")
        size_bytes = model.get("size", 0)
        size_gb = size_bytes / (1024**3)
        print(f"  {i}. {name:<30} ({size_gb:.1f} GB)")
    print("-" * 60)

    while True:
        try:
            choice = input(f"Select model (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
            print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


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
        default=None,
        help="Ollama model to use (interactive menu if not specified)",
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
    parser.add_argument(
        "--list",
        action="store_true",
        help="List installed models and exit",
    )
    args = parser.parse_args()

    # Fetch installed models
    models = get_installed_models(args.base_url)

    # List mode: just show models and exit
    if args.list:
        if models:
            print("\nInstalled models:")
            for model in models:
                name = model.get("name", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                print(f"  {name:<30} ({size_gb:.1f} GB)")
        else:
            print("No models installed.")
        return

    # Select model interactively if not provided
    model = args.model
    if model is None:
        model = select_model_interactive(models)
        if model is None:
            return

    print(f"\nTesting OllamaAgent with model: {model}")
    print(f"Base URL: {args.base_url}")

    agent = OllamaAgent(
        model=model,
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
        print(f"  2. Is the model pulled? Try: ollama pull {model}")
        print(f"  3. Is the server accessible at {args.base_url}?")
        raise


if __name__ == "__main__":
    main()
