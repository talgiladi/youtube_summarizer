import json
import logging
import os
import re
import time
from pathlib import Path

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CONFIG_PATH = Path(__file__).with_name("openai_config.json")
DEFAULT_MAX_TOKENS = 900
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/embedding-001"
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
SUMMARY_CHUNK_SIZE = 4000
SUMMARY_CHUNK_OVERLAP = 400

fetched_transcript = None
processed_transcript = ""


def setup_logging():
    log_level_name = os.getenv("YT_LOG_LEVEL", "DEBUG").upper()
    log_level = getattr(logging, log_level_name, logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("ytbot")
    logger.setLevel(log_level)
    logger.debug("Logging initialized with level=%s", logging.getLevelName(log_level))
    return logger


logger = setup_logging()


def _truncate_for_log(value, limit=160):
    if value is None:
        return ""

    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, total={len(text)} chars]"


def _mask_secret(value):
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


class SimpleLLMChain:
    def __init__(self, llm, prompt, verbose=True):
        self.llm = llm
        self.prompt = prompt
        self.verbose = verbose

    def predict(self, **kwargs):
        start_time = time.perf_counter()
        formatted_prompt = self.prompt.format(**kwargs)
        if self.verbose:
            logger.debug(
                "LLM chain starting | prompt_length=%s | keys=%s | preview=%s",
                len(formatted_prompt),
                sorted(kwargs.keys()),
                _truncate_for_log(formatted_prompt),
            )
        response = self.llm.invoke(formatted_prompt)
        content = getattr(response, "content", response)
        if self.verbose:
            logger.debug(
                "LLM chain completed | response_length=%s | elapsed=%.2fs | preview=%s",
                len(str(content)),
                time.perf_counter() - start_time,
                _truncate_for_log(content),
            )
        return content


def get_transcript(url):
    logger.info("Starting transcript fetch | url=%s", url)
    video_id = get_video_id(url)
    if not video_id:
        logger.warning("Transcript fetch aborted because the URL did not contain a valid video id.")
        return None

    ytt_api = YouTubeTranscriptApi()
    logger.debug("Listing available transcripts | video_id=%s", video_id)
    transcripts = ytt_api.list(video_id)

    transcript = ""
    available_transcripts = []
    for item in transcripts:
        available_transcripts.append(
            {
                "language_code": getattr(item, "language_code", None),
                "is_generated": getattr(item, "is_generated", None),
            }
        )
        if item.language_code == "en":
            if item.is_generated:
                if len(transcript) == 0:
                    logger.debug("Using generated English transcript as a fallback.")
                    transcript = item.fetch()
            else:
                logger.debug("Using manual English transcript.")
                transcript = item.fetch()
                break

    logger.debug("Transcript options discovered | entries=%s", available_transcripts)
    if transcript:
        logger.info("Transcript fetch completed | segments=%s", len(transcript))
    else:
        logger.warning("No usable transcript was found for this video.")
    return transcript if transcript else None


def get_video_id(url):
    pattern = r"https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url or "")
    video_id = match.group(1) if match else None
    logger.debug("Parsed video id | found=%s | video_id=%s", bool(video_id), video_id)
    return video_id


def chunk_transcript(transcript_text, chunk_size=200, chunk_overlap=20):
    logger.debug(
        "Chunking transcript | input_length=%s | chunk_size=%s | chunk_overlap=%s",
        len(transcript_text or ""),
        chunk_size,
        chunk_overlap,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(transcript_text)
    logger.info("Transcript chunking completed | chunks=%s", len(chunks))
    return chunks


def process(transcript):
    if not transcript:
        logger.warning("Transcript processing skipped because no transcript data was provided.")
        return ""

    lines = []
    for item in transcript:
        text = _get_transcript_value(item, "text")
        start = _get_transcript_value(item, "start")
        if text is None or start is None:
            continue
        lines.append(f"Text: {text} Start: {start}")

    processed = "\n".join(lines)
    logger.info(
        "Transcript processing completed | input_segments=%s | output_length=%s",
        len(transcript),
        len(processed),
    )
    return processed


def build_summary_transcript(transcript):
    if not transcript:
        logger.warning("Summary transcript build skipped because no transcript data was provided.")
        return ""

    text_segments = []
    for item in transcript:
        text = _get_transcript_value(item, "text")
        if text:
            text_segments.append(text.strip())

    summary_transcript = "\n".join(segment for segment in text_segments if segment)
    logger.info(
        "Summary transcript prepared | text_segments=%s | output_length=%s",
        len(text_segments),
        len(summary_transcript),
    )
    return summary_transcript


def _get_transcript_value(item, key):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def load_active_provider_config(config_path=CONFIG_PATH):
    logger.debug("Loading active provider config | path=%s", config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    provider_name = config.get("active_provider")
    providers = config.get("providers", {})
    provider_config = providers.get(provider_name)

    if not provider_name or provider_config is None:
        raise ValueError(
            "The config must define 'active_provider' and a matching entry under 'providers'."
        )

    logger.info(
        "Loaded active provider config | active_provider=%s | model=%s",
        provider_name,
        provider_config.get("model"),
    )
    return provider_name, provider_config, providers


def load_embedding_provider_configs(config_path=CONFIG_PATH):
    logger.debug("Loading embedding provider configs | path=%s", config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    embedding_providers = config.get("embedding_providers", {})
    logger.debug(
        "Embedding providers loaded | providers=%s",
        sorted(embedding_providers.keys()),
    )
    return embedding_providers


def setup_credentials(config_path=CONFIG_PATH):
    return load_active_provider_config(config_path)


def build_model_parameters(provider_config):
    parameters = {
        "temperature": provider_config.get("temperature", 0.2),
        "max_tokens": provider_config.get("max_tokens", DEFAULT_MAX_TOKENS),
    }

    if provider_config.get("top_p") is not None:
        parameters["top_p"] = provider_config["top_p"]
    if provider_config.get("frequency_penalty") is not None:
        parameters["frequency_penalty"] = provider_config["frequency_penalty"]
    if provider_config.get("presence_penalty") is not None:
        parameters["presence_penalty"] = provider_config["presence_penalty"]

    logger.debug("Built model parameters | parameters=%s", parameters)
    return parameters


def define_parameters(provider_config):
    return build_model_parameters(provider_config)


def _get_env_api_key(provider_name):
    if provider_name == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider_name == "groq":
        return os.getenv("GROQ_API_KEY")
    if provider_name == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return None


def _is_placeholder_secret(value):
    if not value:
        return True

    normalized = str(value).strip().lower()
    return normalized.startswith("your_") or normalized.endswith("_here")


def resolve_api_key(provider_name, provider_config):
    config_api_key = provider_config.get("api_key")
    if config_api_key and not _is_placeholder_secret(config_api_key):
        logger.debug(
            "Resolved API key from config | provider=%s | key=%s",
            provider_name,
            _mask_secret(config_api_key),
        )
        return config_api_key

    env_api_key = _get_env_api_key(provider_name)
    if env_api_key:
        logger.debug(
            "Resolved API key from environment | provider=%s | key=%s",
            provider_name,
            _mask_secret(env_api_key),
        )
        return env_api_key

    raise ValueError(
        f"Missing a valid API key for provider '{provider_name}' in "
        f"{CONFIG_PATH.name} or the corresponding environment variable."
    )


def _normalize_base_url(provider_name, api_base):
    if not api_base:
        return None

    normalized = str(api_base).rstrip("/")
    if provider_name == "groq" and normalized.endswith("/openai/v1"):
        return normalized[: -len("/openai/v1")]

    return normalized


def initialize_llm(provider_name, provider_config, parameters):
    logger.info(
        "Initializing chat model | provider=%s | model=%s",
        provider_name,
        provider_config.get("model"),
    )
    api_key = resolve_api_key(provider_name, provider_config)
    model_name = provider_config.get("model")

    if not model_name:
        raise ValueError(
            f"Provider '{provider_name}' is missing a 'model' value in {CONFIG_PATH.name}."
        )

    if provider_name == "openai":
        llm_kwargs = {
            "model": model_name,
            "api_key": api_key,
            "temperature": parameters.get("temperature", 0.2),
            "max_tokens": parameters.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        if provider_config.get("api_base"):
            llm_kwargs["base_url"] = _normalize_base_url(
                provider_name, provider_config["api_base"]
            )
        logger.debug(
            "ChatOpenAI kwargs prepared | kwargs=%s",
            {**llm_kwargs, "api_key": _mask_secret(api_key)},
        )
        return ChatOpenAI(**llm_kwargs)

    if provider_name == "groq":
        llm_kwargs = {
            "model": model_name,
            "api_key": api_key,
            "temperature": parameters.get("temperature", 0.2),
            "max_tokens": parameters.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        if provider_config.get("api_base"):
            llm_kwargs["base_url"] = _normalize_base_url(
                provider_name, provider_config["api_base"]
            )
        logger.debug(
            "ChatGroq kwargs prepared | kwargs=%s",
            {**llm_kwargs, "api_key": _mask_secret(api_key)},
        )
        return ChatGroq(**llm_kwargs)

    if provider_name == "gemini":
        llm_kwargs = {
            "model": model_name,
            "google_api_key": api_key,
            "temperature": parameters.get("temperature", 0.2),
            "max_output_tokens": parameters.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        logger.debug(
            "ChatGoogleGenerativeAI kwargs prepared | kwargs=%s",
            {**llm_kwargs, "google_api_key": _mask_secret(api_key)},
        )
        return ChatGoogleGenerativeAI(**llm_kwargs)

    raise ValueError(
        f"Unsupported provider '{provider_name}'. Choose from: openai, groq, gemini."
    )


def _resolve_embedding_provider(active_provider, active_provider_config, providers):
    embedding_provider = active_provider_config.get("embedding_provider", active_provider)
    embedding_providers = load_embedding_provider_configs()
    logger.info(
        "Resolving embedding provider | active_provider=%s | requested_embedding_provider=%s",
        active_provider,
        embedding_provider,
    )

    if embedding_provider in {"openai", "gemini"}:
        embedding_config = dict(embedding_providers.get(embedding_provider, {}))
        embedding_config.update(providers.get(embedding_provider, {}))
        return embedding_provider, embedding_config

    if embedding_provider in {"huggingface", "ollama"}:
        return embedding_provider, embedding_providers.get(embedding_provider, {})

    if embedding_provider == "groq":
        raise ValueError(
            "Groq does not have a configured embedding integration here. "
            "Set 'embedding_provider' to 'openai', 'gemini', 'huggingface', or 'ollama' in openai_config.json."
        )

    if active_provider == "groq":
        if embedding_providers.get("huggingface"):
            return "huggingface", embedding_providers["huggingface"]
        if embedding_providers.get("ollama"):
            return "ollama", embedding_providers["ollama"]

        for fallback_provider in ("openai", "gemini"):
            fallback_config = dict(embedding_providers.get(fallback_provider, {}))
            fallback_config.update(providers.get(fallback_provider, {}))
            try:
                resolve_api_key(fallback_provider, fallback_config)
                return fallback_provider, fallback_config
            except ValueError:
                continue

        raise ValueError(
            "Groq can be used as the chat model, but embeddings need a separate provider. "
            "Set 'embedding_provider' to 'huggingface', 'ollama', 'openai', or 'gemini' in openai_config.json."
        )

    raise ValueError(
        f"Unsupported embedding provider '{embedding_provider}'. "
        "Choose from: openai, gemini, huggingface, ollama."
    )


def initialize_embedding_model(active_provider, active_provider_config, providers):
    embedding_provider, embedding_config = _resolve_embedding_provider(
        active_provider, active_provider_config, providers
    )
    logger.info("Initializing embedding model | provider=%s", embedding_provider)

    if embedding_provider == "openai":
        api_key = resolve_api_key(embedding_provider, embedding_config)
        embedding_model = (
            active_provider_config.get("embedding_model")
            or embedding_config.get("embedding_model")
            or DEFAULT_OPENAI_EMBEDDING_MODEL
        )
        kwargs = {"model": embedding_model, "api_key": api_key}
        if embedding_config.get("api_base"):
            kwargs["base_url"] = embedding_config["api_base"]
        logger.debug(
            "OpenAI embeddings kwargs prepared | kwargs=%s",
            {**kwargs, "api_key": _mask_secret(api_key)},
        )
        return OpenAIEmbeddings(**kwargs)

    if embedding_provider == "gemini":
        api_key = resolve_api_key(embedding_provider, embedding_config)
        embedding_model = (
            active_provider_config.get("embedding_model")
            or embedding_config.get("embedding_model")
            or DEFAULT_GEMINI_EMBEDDING_MODEL
        )
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
        )

    if embedding_provider == "huggingface":
        embedding_model = (
            active_provider_config.get("embedding_model")
            or embedding_config.get("model")
            or embedding_config.get("embedding_model")
            or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
        )
        model_kwargs = embedding_config.get("model_kwargs") or {}
        encode_kwargs = embedding_config.get("encode_kwargs") or {}
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    if embedding_provider == "ollama":
        embedding_model = (
            active_provider_config.get("embedding_model")
            or embedding_config.get("model")
            or embedding_config.get("embedding_model")
            or DEFAULT_OLLAMA_EMBEDDING_MODEL
        )
        ollama_kwargs = {"model": embedding_model}
        if embedding_config.get("base_url"):
            ollama_kwargs["base_url"] = embedding_config["base_url"]
        logger.debug("Ollama embeddings kwargs prepared | kwargs=%s", ollama_kwargs)
        return OllamaEmbeddings(**ollama_kwargs)

    raise ValueError(
        f"Unsupported embedding provider '{embedding_provider}'. "
        "Choose from: openai, gemini, huggingface, ollama."
    )


def setup_embedding_model(credentials=None, project_id=None):
    provider_name, provider_config, providers = load_active_provider_config()
    logger.debug(
        "Setting up embedding model | override_credentials=%s | override_provider=%s",
        isinstance(credentials, dict),
        project_id,
    )

    if isinstance(credentials, dict):
        provider_config = credentials
    if project_id:
        provider_name = project_id
        provider_config = providers.get(provider_name, provider_config)

    return initialize_embedding_model(provider_name, provider_config, providers)


def create_vector_index(chunks, embedding_model):
    start_time = time.perf_counter()
    logger.info("Creating vector index | chunks=%s", len(chunks))
    vector_index = InMemoryVectorStore.from_texts(chunks, embedding_model)
    logger.info(
        "Vector index created | chunks=%s | elapsed=%.2fs",
        len(chunks),
        time.perf_counter() - start_time,
    )
    return vector_index


def perform_similarity_search(vector_index, query, k=3):
    logger.debug("Running similarity search | k=%s | query=%s", k, _truncate_for_log(query))
    results = vector_index.similarity_search(query, k=k)
    logger.debug("Similarity search completed | matches=%s", len(results))
    return results


def create_summary_prompt():
    template = """
    You are an AI assistant tasked with summarizing YouTube video transcripts.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore timestamps in your summary.
    3. Focus only on the spoken content of the video.

    Transcript:
    {transcript}
    """

    return PromptTemplate(input_variables=["transcript"], template=template)


def create_chunk_summary_prompt():
    template = """
    You are an AI assistant tasked with summarizing part of a YouTube video transcript.

    Instructions:
    1. Summarize the transcript section in 2 to 4 sentences.
    2. Keep the main points, decisions, examples, and conclusions.
    3. Ignore timestamps, filler words, and repetition.

    Transcript section:
    {transcript}
    """

    return PromptTemplate(input_variables=["transcript"], template=template)


def create_combined_summary_prompt():
    template = """
    You are an AI assistant tasked with combining partial summaries of a YouTube video.

    Instructions:
    1. Write one concise paragraph that captures the full video.
    2. Preserve the key ideas and overall flow.
    3. Remove repetition and keep the answer readable.

    Partial summaries:
    {transcript}
    """

    return PromptTemplate(input_variables=["transcript"], template=template)


def create_summary_chain(llm, prompt, verbose=True):
    return SimpleLLMChain(llm=llm, prompt=prompt, verbose=verbose)


def retrieve(query, vector_index, k=7):
    logger.info("Retrieving relevant chunks | k=%s | question=%s", k, _truncate_for_log(query))
    results = vector_index.similarity_search(query, k=k)
    logger.info("Retrieval completed | matches=%s", len(results))
    return results


def create_qa_prompt_template():
    qa_template = """
    You are an expert assistant answering questions about a YouTube video.

    Relevant video context:
    {context}

    Question:
    {question}

    Answer using only the context above. If the answer is not in the context, say so clearly.
    """

    return PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template,
    )


def create_qa_chain(llm, prompt_template, verbose=True):
    return SimpleLLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def _format_retrieved_context(relevant_context):
    return "\n\n".join(doc.page_content for doc in relevant_context)


def summarize_transcript_with_fallback(llm, transcript_text):
    if not transcript_text:
        logger.warning("Summarization skipped because transcript text was empty.")
        return ""

    logger.info("Starting transcript summarization | transcript_length=%s", len(transcript_text))
    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)

    if len(transcript_text) <= SUMMARY_CHUNK_SIZE:
        logger.debug("Using single-pass summarization path.")
        return summary_chain.predict(transcript=transcript_text)

    logger.info("Using multi-pass summarization fallback for long transcript.")
    chunk_prompt = create_chunk_summary_prompt()
    chunk_chain = create_summary_chain(llm, chunk_prompt)
    transcript_chunks = chunk_transcript(
        transcript_text,
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
    )
    partial_summaries = [
        chunk_chain.predict(transcript=chunk)
        for chunk in transcript_chunks
    ]
    logger.info("Chunk summaries created | chunks=%s", len(partial_summaries))

    combined_prompt = create_combined_summary_prompt()
    combined_chain = create_summary_chain(llm, combined_prompt)
    combined_input = "\n\n".join(partial_summaries)

    if len(combined_input) <= SUMMARY_CHUNK_SIZE:
        logger.debug("Combined partial summaries fit in a single combine pass.")
        return combined_chain.predict(transcript=combined_input)

    condensed_summaries = chunk_transcript(
        combined_input,
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
    )
    second_pass = [
        combined_chain.predict(transcript=chunk)
        for chunk in condensed_summaries
    ]
    logger.info("Performed second-pass summary condensation | chunks=%s", len(second_pass))
    return combined_chain.predict(transcript="\n\n".join(second_pass))


def generate_answer(question, vector_index, qa_chain, k=7):
    relevant_context = retrieve(question, vector_index, k=k)
    formatted_context = _format_retrieved_context(relevant_context)
    logger.debug(
        "Generating answer from retrieved context | context_length=%s",
        len(formatted_context),
    )
    return qa_chain.predict(context=formatted_context, question=question)


def summarize_video(video_url):
    global fetched_transcript, processed_transcript

    request_start = time.perf_counter()
    logger.info("Summarize request received | url=%s", video_url)
    if not video_url:
        logger.warning("Summarize request rejected because no video URL was provided.")
        return "Please provide a valid YouTube URL."

    fetched_transcript = get_transcript(video_url)
    processed_transcript = process(fetched_transcript)

    if not processed_transcript:
        logger.warning("Summarize request could not continue because the transcript is empty.")
        return "No transcript available. Please fetch the transcript first."

    try:
        provider_name, provider_config, _providers = setup_credentials()
        llm = initialize_llm(
            provider_name,
            provider_config,
            define_parameters(provider_config),
        )
    except Exception as exc:
        logger.exception("Model setup failed during summarization.")
        return f"Model configuration error: {exc}"

    summary_transcript = build_summary_transcript(fetched_transcript)
    if not summary_transcript:
        summary_transcript = processed_transcript

    try:
        summary = summarize_transcript_with_fallback(llm, summary_transcript)
        logger.info(
            "Summarize request completed | summary_length=%s | elapsed=%.2fs",
            len(summary),
            time.perf_counter() - request_start,
        )
        return summary
    except Exception as exc:
        logger.exception("Summary generation failed.")
        return f"Summary generation error: {exc}"


def answer_question(video_url, user_question):
    global fetched_transcript, processed_transcript

    request_start = time.perf_counter()
    logger.info(
        "Question request received | url=%s | question=%s",
        video_url,
        _truncate_for_log(user_question),
    )
    if not processed_transcript:
        if not video_url:
            logger.warning("Question request rejected because no video URL was provided.")
            return "Please provide a valid YouTube URL."
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)

    if not processed_transcript or not user_question:
        logger.warning("Question request rejected because transcript or question is missing.")
        return "Please provide a valid question and ensure the transcript has been fetched."

    chunks = chunk_transcript(processed_transcript)

    try:
        provider_name, provider_config, providers = setup_credentials()
        parameters = define_parameters(provider_config)
        llm = initialize_llm(provider_name, provider_config, parameters)
        embedding_model = initialize_embedding_model(
            provider_name,
            provider_config,
            providers,
        )
    except Exception as exc:
        logger.exception("Model setup failed during question answering.")
        return f"Model configuration error: {exc}"

    vector_index = create_vector_index(chunks, embedding_model)
    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)
    answer = generate_answer(user_question, vector_index, qa_chain)
    logger.info(
        "Question request completed | answer_length=%s | elapsed=%.2fs",
        len(answer),
        time.perf_counter() - request_start,
    )
    return answer


def create_interface():
    logger.debug("Creating Gradio interface.")
    with gr.Blocks() as interface:
        video_url = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter the YouTube Video URL",
        )
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(
            label="Ask a Question About the Video",
            placeholder="Ask your question",
        )
        answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

        summarize_btn = gr.Button("Summarize Video")
        question_btn = gr.Button("Ask a Question")

        summarize_btn.click(
            summarize_video,
            inputs=video_url,
            outputs=summary_output,
        )
        question_btn.click(
            answer_question,
            inputs=[video_url, question_input],
            outputs=answer_output,
        )

    return interface


def main():
    interface = create_interface()
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7862"))
    logger.info(
        "Launching Gradio interface | server_name=%s | server_port=%s",
        server_name,
        server_port,
    )
    interface.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()
