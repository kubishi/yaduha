"""Yaduha Streamlit Dashboard — explore languages, schemas, and translate."""

import inspect
import textwrap
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, get_args, get_origin

import streamlit as st
from pydantic import BaseModel

from yaduha.loader import LanguageLoader
from yaduha.language.language import Language
from yaduha.language import Sentence

# ---------------------------------------------------------------------------
# Page config + compact CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Yaduha", page_icon="Y", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.5rem; }
    h3 { margin-top: 0.5rem !important; margin-bottom: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["Languages", "Translate"], index=0)

# ---------------------------------------------------------------------------
# Helpers — schema to Graphviz DOT
# ---------------------------------------------------------------------------

def _resolve_ref(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _schema_to_dot(schema: Dict[str, Any]) -> str:
    """Convert a JSON Schema (from model_json_schema) to Graphviz DOT."""
    defs = schema.get("$defs", {})
    lines: List[str] = [
        "digraph {",
        '  rankdir=LR;',
        '  node [shape=record, fontsize=10, fontname="Helvetica"];',
        '  edge [fontsize=9, fontname="Helvetica"];',
    ]
    visited: set = set()

    def _node_id(name: str) -> str:
        return name.replace(" ", "_").replace("-", "_")

    def _field_label(name: str, prop: Dict[str, Any]) -> str:
        if "$ref" in prop:
            return _resolve_ref(prop["$ref"])
        if "anyOf" in prop:
            parts = []
            for item in prop["anyOf"]:
                if item.get("type") == "null":
                    continue
                if "$ref" in item:
                    parts.append(_resolve_ref(item["$ref"]))
                elif "type" in item:
                    parts.append(item["type"])
            result = " | ".join(parts) if parts else "?"
            has_null = any(i.get("type") == "null" for i in prop["anyOf"])
            return result + "?" if has_null else result
        if "enum" in prop:
            return "enum"
        return prop.get("type", "?")

    def _add_model_node(name: str, props: Dict[str, Any], color: str, required: List[str]):
        nid = _node_id(name)
        if nid in visited:
            return
        visited.add(nid)
        field_rows = []
        for fname, fprop in props.items():
            opt = "" if fname in required else "?"
            ftype = _field_label(fname, fprop)
            field_rows.append(f"<{fname}> {fname}{opt}: {ftype}")
        label = "{" + name + "|" + "\\l".join(field_rows) + "\\l}"
        lines.append(f'  {nid} [label="{label}", style=filled, fillcolor="{color}"];')

    def _add_enum_node(name: str, values: List):
        nid = _node_id(name)
        if nid in visited:
            return
        visited.add(nid)
        vals = "\\l".join(str(v) for v in values) + "\\l"
        label = "{" + name + "|" + vals + "}"
        lines.append(f'  {nid} [label="{label}", style=filled, fillcolor="#fef3c7"];')

    def _add_edges(parent_name: str, props: Dict[str, Any]):
        pid = _node_id(parent_name)
        for fname, fprop in props.items():
            targets = []
            if "$ref" in fprop:
                targets.append(_resolve_ref(fprop["$ref"]))
            elif "anyOf" in fprop:
                for item in fprop["anyOf"]:
                    if "$ref" in item:
                        targets.append(_resolve_ref(item["$ref"]))
            for target in targets:
                tid = _node_id(target)
                lines.append(f'  {pid}:{fname} -> {tid};')

    root_name = schema.get("title", "Root")
    root_props = schema.get("properties", {})
    root_required = schema.get("required", [])
    _add_model_node(root_name, root_props, "#dbeafe", root_required)
    _add_edges(root_name, root_props)

    for def_name, def_schema in defs.items():
        if "enum" in def_schema:
            _add_enum_node(def_name, def_schema["enum"])
        elif "properties" in def_schema:
            def_props = def_schema["properties"]
            def_required = def_schema.get("required", [])
            _add_model_node(def_name, def_props, "#d1fae5", def_required)
            _add_edges(def_name, def_props)

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers — generic component discovery
# ---------------------------------------------------------------------------

def _discover_components(sentence_cls: Type[Sentence]) -> Dict[str, Any]:
    """Walk model_fields annotations to find all referenced classes and module functions."""
    classes: Dict[str, type] = {}

    def _walk_annotation(ann: Any) -> None:
        origin = get_origin(ann)
        args = get_args(ann)
        if isinstance(ann, type) and issubclass(ann, (BaseModel, Enum)):
            if ann.__name__ not in classes:
                classes[ann.__name__] = ann
                if issubclass(ann, BaseModel):
                    for field_info in ann.model_fields.values():
                        _walk_annotation(field_info.annotation)
        if args:
            for arg in args:
                _walk_annotation(arg)

    classes[sentence_cls.__name__] = sentence_cls
    for field_info in sentence_cls.model_fields.values():
        _walk_annotation(field_info.annotation)

    # Module-level functions from the sentence's defining module
    module = inspect.getmodule(sentence_cls)
    functions: Dict[str, Callable] = {}
    if module:
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if getattr(func, "__module__", None) == module.__name__:
                functions[name] = func

    return {"classes": classes, "functions": functions}


_PYDANTIC_SKIP = {
    "copy", "dict", "json", "model_copy", "model_dump", "model_dump_json",
    "model_post_init", "model_validate", "model_validate_json", "parse_obj",
    "parse_raw", "schema", "validate", "construct", "from_orm",
    "update_forward_refs", "model_rebuild", "model_json_schema",
    "model_parametrized_name", "model_construct",
}


def _get_interesting_methods(cls: type) -> List[Tuple[str, Any]]:
    """Return methods defined directly on cls, filtering out Pydantic boilerplate."""
    results = []
    for name in sorted(cls.__dict__):
        if name in _PYDANTIC_SKIP:
            continue
        if name.startswith("_") and name != "__str__":
            continue
        # Skip __str__ inherited from str on str-based Enums (not useful to show)
        if name == "__str__" and issubclass(cls, (str, int)) and issubclass(cls, Enum):
            continue
        obj = cls.__dict__[name]
        # Accept regular functions, classmethods, staticmethods
        if callable(obj) or isinstance(obj, (classmethod, staticmethod)):
            results.append((name, obj))
    return results


def _build_source_options(
    sentence_cls: Type[Sentence],
    components: Dict[str, Any],
) -> List[Tuple[str, Any, Optional[str]]]:
    """Build (label, target, method_name) tuples for the source code selectbox."""
    options: List[Tuple[str, Any, Optional[str]]] = []

    # Sentence class methods first
    for method_name, _ in _get_interesting_methods(sentence_cls):
        options.append((f"{sentence_cls.__name__}.{method_name}()", sentence_cls, method_name))

    # Other classes
    for cls_name, cls in components["classes"].items():
        if cls is sentence_cls:
            continue
        methods = _get_interesting_methods(cls)
        for method_name, _ in methods:
            options.append((f"{cls_name}.{method_name}()", cls, method_name))
        # Full class source
        options.append((f"{cls_name} (full class)", cls, None))

    # Module-level functions
    for func_name, func in components["functions"].items():
        options.append((f"{func_name}()", func, None))

    return options


# ---------------------------------------------------------------------------
# Helpers — rendering trace
# ---------------------------------------------------------------------------

def _collect_intermediates(
    instance: Sentence,
    sentence_cls: Type[Sentence],
) -> List[Tuple[str, str, str]]:
    """Collect intermediate rendering values from an instance.

    Returns list of (call_expression, result, method_source) tuples.
    """
    module = inspect.getmodule(sentence_cls)
    results: List[Tuple[str, str, str]] = []
    seen: set = set()

    def _try_get_source(obj: Any) -> str:
        try:
            return textwrap.dedent(inspect.getsource(obj))
        except (OSError, TypeError):
            return ""

    def _call_zero_arg_methods(path: str, value: Any, cls: type) -> None:
        for name in sorted(dir(value)):
            if not name.startswith("get_"):
                continue
            method = getattr(value, name, None)
            if method is None or not callable(method):
                continue
            try:
                sig = inspect.signature(method)
            except (ValueError, TypeError):
                continue
            if any(
                p.default is inspect.Parameter.empty
                for p in sig.parameters.values()
            ):
                continue
            call_key = f"{path}.{name}"
            if call_key in seen:
                continue
            seen.add(call_key)
            try:
                result = method()
                source = _try_get_source(getattr(cls, name))
                results.append((f"{call_key}()", repr(result), source))
            except Exception:
                pass

    def _try_vocab_lookup(path: str, attr_name: str, value: str, func_names: List[str]) -> None:
        if not module:
            return
        for fn_name in func_names:
            func = getattr(module, fn_name, None)
            if func is None:
                continue
            call_key = f'{fn_name}("{value}")'
            if call_key in seen:
                continue
            try:
                result = func(value)
                seen.add(call_key)
                source = _try_get_source(func)
                results.append((call_key, repr(result), source))
                return
            except (KeyError, ValueError):
                continue

    def _process(path: str, value: Any) -> None:
        if isinstance(value, BaseModel):
            for fname, fval in value:
                _process(f"{path}.{fname}", fval)
            _call_zero_arg_methods(path, value, type(value))
            # Vocab lookups
            if hasattr(value, "lemma") and isinstance(value.lemma, str):
                _try_vocab_lookup(
                    path, "lemma", value.lemma,
                    ["get_verb_target", "get_transitive_verb_target",
                     "get_intransitive_verb_target"],
                )
            if hasattr(value, "head") and isinstance(value.head, str):
                _try_vocab_lookup(path, "head", value.head, ["get_noun_target"])
        elif isinstance(value, Enum):
            _call_zero_arg_methods(path, value, type(value))

    for fname, fval in instance:
        _process(fname, fval)

    return results


def _render_trace(instance: Sentence, sentence_cls: Type[Sentence]) -> None:
    """Display a step-by-step rendering trace for a sentence instance."""
    st.code(str(instance), language=None)

    try:
        str_source = textwrap.dedent(inspect.getsource(sentence_cls.__str__))
    except OSError:
        str_source = "# source not available"

    with st.expander("__str__ source", expanded=True):
        st.code(str_source, language="python")

    intermediates = _collect_intermediates(instance, sentence_cls)
    if not intermediates:
        return

    st.markdown("**Intermediate values**")
    for i, (call_expr, result, source) in enumerate(intermediates, 1):
        with st.expander(f"`{call_expr}` = `{result}`"):
            if source:
                st.code(source, language="python")


def _render_structured_view(instance: Sentence) -> None:
    """Compact field-by-field view of a sentence instance."""
    for field_name, field_value in instance:
        if isinstance(field_value, BaseModel):
            st.markdown(f"**{field_name}** (`{type(field_value).__name__}`)")
            sub_items = list(field_value)
            if sub_items:
                cols = st.columns(min(len(sub_items), 4))
                for col, (sub_name, sub_val) in zip(cols, sub_items):
                    with col:
                        display = sub_val.value if isinstance(sub_val, Enum) else sub_val
                        st.caption(sub_name)
                        st.code(str(display), language=None)
                # Handle overflow if more than 4 sub-fields
                if len(sub_items) > 4:
                    cols2 = st.columns(min(len(sub_items) - 4, 4))
                    for col, (sub_name, sub_val) in zip(cols2, sub_items[4:]):
                        with col:
                            display = sub_val.value if isinstance(sub_val, Enum) else sub_val
                            st.caption(sub_name)
                            st.code(str(display), language=None)
        elif isinstance(field_value, Enum):
            st.markdown(f"**{field_name}**: `{field_value.value}`")
        else:
            st.markdown(f"**{field_name}**: `{field_value}`")


# ---------------------------------------------------------------------------
# Helpers — agent creation
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": {
        "module": "yaduha.agent.openai",
        "class": "OpenAIAgent",
        "env_var": "OPENAI_API_KEY",
        "needs_key": True,
    },
    "anthropic": {
        "module": "yaduha.agent.anthropic",
        "class": "AnthropicAgent",
        "env_var": "ANTHROPIC_API_KEY",
        "needs_key": True,
    },
    "gemini": {
        "module": "yaduha.agent.gemini",
        "class": "GeminiAgent",
        "env_var": "GEMINI_API_KEY",
        "needs_key": True,
    },
    "ollama": {
        "module": "yaduha.agent.ollama",
        "class": "OllamaAgent",
        "env_var": None,
        "needs_key": False,
    },
}


def _get_models_for_provider(provider: str) -> List[str]:
    """Extract the Literal model choices from the agent class."""
    import importlib
    info = PROVIDERS[provider]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    ann = cls.model_fields["model"].annotation
    origin = get_origin(ann)
    if origin is Literal:
        return list(get_args(ann))
    return []


def _create_agent(provider: str, model: str, api_key: Optional[str]):
    import importlib
    info = PROVIDERS[provider]
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    kwargs: Dict[str, Any] = {"model": model}
    if info["needs_key"]:
        kwargs["api_key"] = api_key
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Languages page
# ---------------------------------------------------------------------------

def page_languages():
    st.title("Languages")
    languages = LanguageLoader.list_installed_languages()

    if not languages:
        st.info("No languages installed.")
        return

    lang_options = {f"{l.name} ({l.code})": l for l in languages}
    selected_label = st.sidebar.selectbox("Language", list(lang_options.keys()))
    lang = lang_options[selected_label]

    st.header(f"{lang.name}")
    st.caption(f"Code: `{lang.code}` · {len(lang.sentence_types)} sentence type(s)")

    type_names = [st_cls.__name__ for st_cls in lang.sentence_types]
    tabs = st.tabs(type_names)

    for tab, st_cls in zip(tabs, lang.sentence_types):
        with tab:
            _render_sentence_type(st_cls)


def _render_sentence_type(st_cls: Type[Sentence]):
    schema = st_cls.model_json_schema()
    components = _discover_components(st_cls)
    options = _build_source_options(st_cls, components)

    # --- Top row: schema graph + source code explorer ---
    col_graph, col_source = st.columns([2, 3])

    with col_graph:
        st.markdown("##### Schema")
        dot = _schema_to_dot(schema)
        st.graphviz_chart(dot)

    with col_source:
        st.markdown("##### Source Code")

        labels = [opt[0] for opt in options]
        default_idx = next(
            (i for i, lbl in enumerate(labels) if lbl == f"{st_cls.__name__}.__str__()"),
            0,
        )

        selected_label = st.selectbox(
            "Inspect",
            labels,
            index=default_idx,
            key=f"source_{st_cls.__name__}",
            label_visibility="collapsed",
        )

        _, target, method_name = options[labels.index(selected_label)]

        try:
            if method_name:
                raw = inspect.getsource(getattr(target, method_name))
            else:
                raw = inspect.getsource(target)
            source = textwrap.dedent(raw)
        except OSError:
            source = "# source not available"

        st.code(source, language="python", line_numbers=True)

        with st.expander("JSON Schema"):
            st.json(schema)

    # --- Bottom row: examples ---
    st.markdown("##### Examples")
    examples = st_cls.get_examples()
    if not examples:
        st.info("No examples defined for this sentence type.")
        return

    for i, (english, instance) in enumerate(examples):
        with st.expander(
            f'"{english}" \u2192 **{str(instance)}**',
            expanded=(i == 0),
        ):
            tab_trace, tab_struct, tab_json = st.tabs(
                ["Rendering Trace", "Structured", "JSON"]
            )
            with tab_trace:
                _render_trace(instance, st_cls)
            with tab_struct:
                _render_structured_view(instance)
            with tab_json:
                st.json(instance.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Translate page
# ---------------------------------------------------------------------------

def page_translate():
    st.title("Translate")

    languages = LanguageLoader.list_installed_languages()
    if not languages:
        st.warning("No languages installed.")
        return

    # API keys in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Keys")
    st.sidebar.caption("Stored in session only")
    for prov, info in PROVIDERS.items():
        if info["needs_key"]:
            key = f"api_key_{prov}"
            if key not in st.session_state:
                st.session_state[key] = ""
            st.session_state[key] = st.sidebar.text_input(
                f"{prov.title()} API Key",
                value=st.session_state[key],
                type="password",
                key=f"input_{key}",
            )

    # Main form
    col_left, col_right = st.columns([1, 1])

    with col_left:
        text = st.text_area("English text", placeholder="Enter text to translate...", height=120)

        lang_map = {f"{l.name} ({l.code})": l for l in languages}
        lang_label = st.selectbox("Language", list(lang_map.keys()))
        lang = lang_map[lang_label]

        provider = st.selectbox("Provider", list(PROVIDERS.keys()), format_func=str.title)
        models = _get_models_for_provider(provider)
        if models:
            model = st.selectbox("Model", models)
        else:
            model = st.text_input("Model", value="llama3.1")

        translate_btn = st.button("Translate", type="primary", disabled=not text.strip())

    with col_right:
        if translate_btn and text.strip():
            api_key = st.session_state.get(f"api_key_{provider}", "")
            if PROVIDERS[provider]["needs_key"] and not api_key:
                st.error(f"Please set your {provider.title()} API key in the sidebar.")
                return

            try:
                agent = _create_agent(provider, model, api_key or None)
            except Exception as e:
                st.error(f"Failed to create agent: {e}")
                return

            from yaduha.translator.pipeline import PipelineTranslator

            try:
                translator = PipelineTranslator(
                    agent=agent,
                    SentenceType=lang.sentence_types,
                )
            except Exception as e:
                st.error(f"Failed to create translator: {e}")
                return

            with st.spinner("Translating..."):
                try:
                    result = translator.translate(text.strip())
                except Exception as e:
                    st.error(f"Translation failed: {e}")
                    return

            st.subheader("Result")

            st.markdown("**Translation**")
            st.code(result.target, language=None)

            st.markdown("**Source**")
            st.write(result.source)

            if result.back_translation:
                st.markdown("**Back-translation**")
                st.write(result.back_translation.source)

            st.caption(
                f"Translation: {result.translation_time:.2f}s · "
                f"{result.prompt_tokens + result.completion_tokens} tokens"
            )
            if result.back_translation:
                st.caption(
                    f"Back-translation: {result.back_translation.translation_time:.2f}s · "
                    f"{result.back_translation.prompt_tokens + result.back_translation.completion_tokens} tokens"
                )

            with st.expander("Raw response"):
                st.json(result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Languages":
    page_languages()
elif page == "Translate":
    page_translate()
