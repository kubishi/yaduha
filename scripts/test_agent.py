from typing import Any, ClassVar, Dict, List, Tuple
from pydantic import BaseModel, Field
from yaduha.tool import Tool
from yaduha.translator.pipeline import PipelineTranslator
from yaduha.agent.openai import OpenAIAgent
from yaduha.agent.claude import ClaudeAgent
from yaduha.language.ovp import SubjectVerbSentence, SubjectVerbObjectSentence

from dotenv import load_dotenv
import os

load_dotenv()

class Person(BaseModel):
    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")

PEOPLE = [
    Person(name="Alice Smith", age=79),
    Person(name="Bob Johnson", age=25),
    Person(name="Charlie Brown", age=35),
]

class Random:
    def __init__(self) -> None:
        pass

class GetPeople(Tool):
    name: ClassVar[str] = "get_people"
    description: ClassVar[str] = "Get a list of people in our game."

    def _run(self) -> list["Person"]:
        print("GetPeople called")
        return PEOPLE
    
class SearchPeople(Tool):
    name: ClassVar[str] = "search_people"
    description: ClassVar[str] = "Search for people in our game."

    def _run(self, person: Person) -> list["Person"]:
        #Is this an error? I'm pretty sure this needs to filter out a person from the list of people
        print(f"SearchPeople called with person={person}")
        results = [
            person for person in PEOPLE 
            if person.name.lower() in person.name.lower()
        ]
        return results

    def get_examples(self) -> List[Tuple[Dict[str, Person], List[Person]]]:
        return [
            ({"person": Person(name="Alice Smith", age=79)}, [Person(name="Alice Smith", age=79)])
        ]
    
    # def _run(self, person_name: str) -> list["Person"]:
    #     print(f"SearchPeople called with person_name={person_name}")
    #     results = [
    #         person for person in PEOPLE 
    #         if person.name.lower() in person_name.lower()
    #     ]
    #     return results
        
    
def main():
    # agent = OpenAIAgent(
    #     model="gpt-4o",
    #     api_key=os.environ["OPENAI_API_KEY"]
    # )

    agent = ClaudeAgent(
        model="claude-sonnet-4-5",
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    response = agent.get_response(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that generates random characters "
                    "for a role-playing game based on user requests. "
                    "Use the provided tools to get information about existing characters "
                    "for consistency and inspiration."
                )
            },
            {
                "role": "user",
                "content": "Generate a person that could be Alice's daughter."
            }
        ],
        response_format=Person,
        tools=[SearchPeople()]
    )

    print("Response Content:", response.content)
    print("Response Time:", response.response_time)
    print("Prompt Tokens:", response.prompt_tokens)
    print("Completion Tokens:", response.completion_tokens)

    

if __name__ == "__main__":
    main()

