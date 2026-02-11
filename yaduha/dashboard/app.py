"""Yaduha Streamlit Dashboard — explore languages, schemas, and translate."""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, get_args, get_origin

import streamlit as st
from pydantic import BaseModel

from yaduha.loader import LanguageLoader
from yaduha.language.language import Language
from yaduha.language import Sentence

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Yaduha", page_icon="Y", layout="wide")

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
        """Summarize a field's type for the node label."""
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

    # Process root
    root_name = schema.get("title", "Root")
    root_props = schema.get("properties", {})
    root_required = schema.get("required", [])
    _add_model_node(root_name, root_props, "#dbeafe", root_required)
    _add_edges(root_name, root_props)

    # Process $defs
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
    return []  # ollama — free-form string


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

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Schema Graph")
        dot = _schema_to_dot(schema)
        st.graphviz_chart(dot)

    with col2:
        st.subheader("JSON Schema")
        st.json(schema)

    st.subheader("Examples")
    examples = st_cls.get_examples()
    if not examples:
        st.info("No examples defined for this sentence type.")
        return

    for english, instance in examples:
        with st.expander(f'"{english}" → {str(instance)}'):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rendered**")
                st.code(str(instance), language=None)
            with c2:
                st.markdown("**Structured**")
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
