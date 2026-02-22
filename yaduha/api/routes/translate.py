"""Routes for performing translations."""

from fastapi import APIRouter, Request

from yaduha.api.dependencies import create_agent, create_evaluator, get_language
from yaduha.api.models import AgenticTranslateRequest, TranslateRequest
from yaduha.tool.english_to_sentences import EnglishToSentencesTool
from yaduha.tool.sentence_to_english import SentenceToEnglishTool
from yaduha.translator import Translation
from yaduha.translator.agentic import AgenticTranslator
from yaduha.translator.pipeline import PipelineTranslator

router = APIRouter(prefix="/translate", tags=["translate"])


def _headers_dict(request: Request) -> dict[str, str]:
    return {k.lower(): v for k, v in request.headers.items()}


@router.post("/pipeline", response_model=Translation)
async def translate_pipeline(body: TranslateRequest, request: Request):
    """Translate text using the pipeline translator.

    The pipeline translator guarantees grammatical output by constraining
    generation to the language's defined sentence structures.
    """
    headers = _headers_dict(request)
    agent = create_agent(body.agent, headers)

    bt_agent = None
    if body.back_translation_agent:
        bt_agent = create_agent(body.back_translation_agent, headers)

    evaluators = []
    if body.evaluators:
        evaluators = [create_evaluator(ec, headers) for ec in body.evaluators]

    lang = get_language(body.language_code)

    translator = PipelineTranslator(
        agent=agent,
        back_translation_agent=bt_agent,
        SentenceType=lang.sentence_types,
        evaluators=evaluators,
    )

    return translator.translate(body.text)


@router.post("/agentic", response_model=Translation)
async def translate_agentic(body: AgenticTranslateRequest, request: Request):
    """Translate text using the agentic translator.

    The agentic translator uses LLM reasoning with tool assistance for
    flexible translation. Confidence levels and evidence are logged.
    """
    headers = _headers_dict(request)
    agent = create_agent(body.agent, headers)
    lang = get_language(body.language_code)

    e2s_tool = EnglishToSentencesTool(agent=agent, SentenceType=lang.sentence_types)
    s2e_tool = SentenceToEnglishTool(agent=agent, SentenceType=lang.sentence_types)
    pipeline_tool = PipelineTranslator(agent=agent, SentenceType=lang.sentence_types)

    kwargs = {"agent": agent, "tools": [e2s_tool, s2e_tool, pipeline_tool]}
    if body.system_prompt:
        kwargs["system_prompt"] = body.system_prompt

    translator = AgenticTranslator(**kwargs)
    return translator.translate(body.text)
