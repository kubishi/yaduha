from functools import lru_cache
import os
from uuid import uuid4
from pydantic import BaseModel, Field, create_model
from typing import Any, ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar, get_origin, get_args, Union
from abc import abstractmethod
import random
import string
import inspect

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from yaduha.logger import Logger, get_global_logger, inject_logs

def _add_additional_properties_false(schema: Dict | List) -> None:
    """Recursively add 'additionalProperties': False to all object schemas."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            _add_additional_properties_false(value)
    elif isinstance(schema, list):
        for item in schema:
            _add_additional_properties_false(item)

_T = TypeVar("_T")
class Tool(BaseModel, Generic[_T]):
    name: ClassVar[str] = Field(..., description="The name of the tool.")
    description: ClassVar[str] = Field(..., description="A description of what the tool does.")
    logger: Logger = Field(default_factory=get_global_logger, description="The logger to use for logging tool actions.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.name.isidentifier():
            raise ValueError(f"Tool name '{self.name}' is not a valid Python identifier.")
        if self.name in {"print", "input", "len", "str", "int", "float", "list", "dict", "set", "tuple"}:
            raise ValueError(f"Tool name '{self.name}' is a reserved Python keyword or built-in function name.")
        
        self._validate_run()
        self._validate_examples()

    def __call__(self, *args, **kwargs) -> _T:
        """Call the tool with the given arguments.

        Automatically parses BaseModel arguments.
        """
        signature = inspect.signature(self._run)
        bound_args = signature.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            param = signature.parameters[name]
            # Check if annotation is a class and a BaseModel before attempting parsing
            if (inspect.isclass(param.annotation) and
                issubclass(param.annotation, BaseModel) and
                not isinstance(value, param.annotation)):
                bound_args.arguments[name] = param.annotation(**value)

        toolchain = os.environ.get("LOGGER_METADATA_TOOLCHAIN", "")
        if not toolchain:
            toolchain = str(uuid4())
        else:
            toolchain = f"{toolchain}/{str(uuid4())}"
        with inject_logs(tool=self.name, toolchain=toolchain):
            return self._run(*bound_args.args, **bound_args.kwargs)

    @abstractmethod
    def _run(self, *args, **kwargs) -> _T:
        pass

    def get_examples(self) -> List[Tuple[Any, _T]]:
        """Return a list of example inputs and outputs for this tool.

        Subclasses can override with more specific types (covariant return types).
        For a tool with `_run(self, query: str) -> Person`, override as:

            def get_examples(self) -> List[Tuple[str, Person]]:
                return [("Alice", Person(name="Alice", age=30))]

        Returns:
            List of (input, expected_output) tuples. Input can be a single
            value or dict of kwargs matching the _run signature.
        """
        return []

    def log(self, data: Dict[str, Any]):
        if self.logger is not None:
            self.logger.log(data)

    @classmethod
    @lru_cache(maxsize=None)
    def _validate_run(cls) -> None:
        """Validate that the _run method conforms to rules:
        - All parameters must have type annotations.
        - Types must be str, int, float, bool, BaseModel, or List/Dict of these types.
        """
        import typing

        def _check_type(annotation: Any) -> bool:
            # Use typing.get_origin and typing.get_args for better generic handling
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Basic types
            if annotation in {str, int, float, bool}:
                return True

            # Handle TypeVar (generic type parameters like TSentenceType)
            if isinstance(annotation, TypeVar):
                # Check if the TypeVar has a bound (e.g., bound=Sentence)
                if annotation.__bound__ is not None:
                    return _check_type(annotation.__bound__)
                # Check if the TypeVar has constraints
                if annotation.__constraints__:
                    return all(_check_type(constraint) for constraint in annotation.__constraints__)
                # If no bound or constraints, accept it (Any-like behavior)
                return True

            # Check if it's a BaseModel class directly
            if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
                return True

            # Handle generic types (e.g., Translation[BackTranslation])
            # The origin will be the base class (e.g., Translation)
            if origin is not None:
                # Check if origin is a BaseModel
                if inspect.isclass(origin) and issubclass(origin, BaseModel):
                    # Validate all type arguments recursively
                    return all(_check_type(arg) for arg in args) if args else True

                # Handle List and Dict generics
                if origin in {list, List}:
                    if len(args) == 1:
                        return _check_type(args[0])
                    elif len(args) == 0:
                        # List without type parameter (e.g., List or list) - accept it
                        return True
                if origin in {dict, Dict}:
                    if len(args) == 2:
                        return _check_type(args[0]) and _check_type(args[1])
                    elif len(args) == 0:
                        # Dict without type parameters (e.g., Dict or dict) - accept it
                        return True

                # Handle Union types (including Optional which is Union[X, None])
                if origin is Union:
                    return all(_check_type(arg) for arg in args if arg is not type(None))

            return False

        # Use typing.get_type_hints to resolve forward references automatically
        try:
            type_hints = typing.get_type_hints(cls._run)
        except Exception:
            # If get_type_hints fails, fall back to using signature
            type_hints = {}
            signature = inspect.signature(cls._run)
            for name, param in signature.parameters.items():
                if param.annotation != inspect._empty:
                    type_hints[name] = param.annotation
            if signature.return_annotation != inspect._empty:
                type_hints['return'] = signature.return_annotation

        # Validate parameters
        signature = inspect.signature(cls._run)
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if name not in type_hints:
                raise ValueError(f"Parameter '{name}' in tool '{cls.name}' has no type annotation.")
            annotation = type_hints[name]
            if not _check_type(annotation):
                raise ValueError(
                    f"Parameter '{name}' in tool '{cls.name}' has unsupported type annotation '{annotation}'. "
                    "Supported types are str, int, float, bool, BaseModel, List, and Dict of these types."
                )

        # Validate return type
        if 'return' not in type_hints:
            raise ValueError(f"Return type of tool '{cls.name}' has no type annotation.")
        return_type = type_hints['return']
        if not _check_type(return_type):
            raise ValueError(
                f"Return type of tool '{cls.name}' has unsupported type annotation '{return_type}'. "
                "Supported types are str, int, float, bool, BaseModel, List, and Dict of these types."
            )

    def _validate_examples(self) -> None:
        """Validate that the examples conform to the _run method signature.

        Validates input and output types.
        """
        import typing

        # Get type hints with forward references resolved
        try:
            type_hints = typing.get_type_hints(self._run)
        except Exception:
            # If that fails, fall back to annotations
            type_hints = {}
            signature = inspect.signature(self._run)
            for name, param in signature.parameters.items():
                if param.annotation != inspect._empty:
                    type_hints[name] = param.annotation
            if signature.return_annotation != inspect._empty:
                type_hints['return'] = signature.return_annotation

        signature = inspect.signature(self._run)
        # Create a new signature without 'self'
        params_without_self = [
            param for name, param in signature.parameters.items()
            if name != 'self'
        ]
        signature_without_self = signature.replace(parameters=params_without_self)

        examples = self.get_examples()
        for i, example in enumerate(examples):
            if len(example) != 2:
                raise ValueError(
                    f"Example {i} in tool '{self.name}' must be a tuple of (input, output), got {len(example)} elements"
                )

            example_input, example_output = example

            # Bind the input to the signature (without self)
            try:
                if isinstance(example_input, dict):
                    bound_args = signature_without_self.bind(**example_input)
                else:
                    # For single argument, bind it directly
                    bound_args = signature_without_self.bind(example_input)
            except TypeError as e:
                raise ValueError(
                    f"Example {i} input in tool '{self.name}' does not match _run signature: {e}"
                )

            # Validate input parameter types
            for name, value in bound_args.arguments.items():
                if name not in type_hints:
                    continue

                expected_type = type_hints[name]
                origin = get_origin(expected_type)

                # Skip validation for TypeVars (they're validated at definition time)
                if isinstance(expected_type, TypeVar):
                    continue

                # For generic types like list[Person], check the origin
                if origin is not None:
                    if not isinstance(value, origin):
                        raise ValueError(
                            f"Example {i} input for parameter '{name}' in tool '{self.name}' "
                            f"is of type {type(value).__name__}, expected {origin.__name__}"
                        )
                elif not isinstance(value, expected_type):
                    raise ValueError(
                        f"Example {i} input for parameter '{name}' in tool '{self.name}' "
                        f"is of type {type(value).__name__}, expected {expected_type.__name__}"
                    )

            # Validate output type
            if 'return' in type_hints:
                return_type = type_hints['return']
                origin = get_origin(return_type)

                # Skip validation for TypeVars (they're validated at definition time)
                if isinstance(return_type, TypeVar):
                    continue

                # For generic types like list[Person], check the origin
                if origin is not None:
                    if not isinstance(example_output, origin):
                        raise ValueError(
                            f"Example {i} output in tool '{self.name}' is of type {type(example_output).__name__}, "
                            f"expected {origin.__name__}"
                        )
                elif not isinstance(example_output, return_type):
                    raise ValueError(
                        f"Example {i} output in tool '{self.name}' is of type {type(example_output).__name__}, "
                        f"expected {return_type.__name__}"
                    )

    @staticmethod
    def get_random_tool_call_id():
        """Generate a random tool call id of the form call_aSENunZCF31ob7zV89clvL4n"""
        return "call_" + ''.join(random.choices(string.ascii_letters + string.digits, k=24))

    def get_tool_call_schema(self) -> ChatCompletionToolParam:
        signature = inspect.signature(self._run)
        properties = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            annotation = param.annotation
            if annotation == inspect._empty:
                annotation = Any
            default = param.default
            if default == inspect._empty:
                default = ...
            properties[name] = (annotation, default)

        model = create_model(self.name, **properties)
        parameters_schema = model.model_json_schema()
        _add_additional_properties_false(parameters_schema)

        if "properties" in parameters_schema:
            parameters_schema["required"] = list(parameters_schema.get("properties", {}).keys())

        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters_schema,
                "strict": True,
            }
        }
        return ChatCompletionToolParam(**schema)
    
    def get_tool_call_output_schema(self) -> Dict:
        signature = inspect.signature(self._run)
        return_type = signature.return_annotation
        if return_type == inspect._empty:
            return_type = Any
        model = create_model(f"{self.name}_output", output=(return_type, ...))
        schema = model.model_json_schema()
        return schema
    

