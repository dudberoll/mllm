from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence
from unittest import result

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import build_reference_config, ensure_directories, seed_everything
import embedding as embedding_module
from embedding import SentenceEmbedder
from preprocessing import PyMuPDFLoader
from vector_db import VectorDB


DEFAULT_QUESTIONS = (
    "What was the first Russian state?",
    "Who was the last king of Poland?",
    "Who does rule Russia currently?",
    "How many people did Russia lose because of the separate peace in 1918?",
    "What is capital of Finland?",
)


@dataclass
class GenerationResult:
    answer: str
    contexts: list[str]
    scores: list[float]
    augmented_prompt: str


def default_answers_path() -> Path:
    return Path(__file__).resolve().parent / "answers.json"


def save_answers(
    questions: Sequence[str],
    answers: Sequence[str],
    output_path: Path | None = None,
) -> Path:
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have the same length")

    target_path = Path(output_path) if output_path is not None else default_answers_path()
    payload = [
        {"question": str(question), "answer": str(answer)}
        for question, answer in zip(questions, answers, strict=True)
    ]
    target_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target_path


def load_answers(output_path: Path | None = None) -> dict[str, str]:
    source_path = Path(output_path) if output_path is not None else default_answers_path()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("answers.json must contain a JSON list")

    loaded_answers: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each answers.json entry must be a JSON object")
        if "question" not in item or "answer" not in item:
            raise ValueError("Each answers.json entry must contain question and answer")

        loaded_answers[str(item["question"])] = str(item["answer"])

    return loaded_answers


class RAGGenerator:
    """Retrieve context from the vector DB and synthesize a final answer."""

    def __init__(
        self,
        db: VectorDB,
        model_name: str | None = None,
        threshold: float = 0.3,
        top_n: int = 3,
        use_rerank: bool = True,
        max_new_tokens: int = 128,
        fallback_answer: str = "Can't anwer with provided info",
        base_system_prompt: str | None = None,
    ) -> None:
        self.db = db
        self.model_name = model_name
        self.threshold = threshold
        self.top_n = top_n
        self.use_rerank = use_rerank
        self.max_new_tokens = max_new_tokens
        self.fallback_answer = fallback_answer
        self.base_system_prompt = base_system_prompt or (
            "TODO"
        )
        self.device = "cpu"
        self.model = None
        self.tokenizer = None

        if self.model_name:
            self._load_model()

    def _load_model(self) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def augment(self, query: str, contexts: Sequence[str]) -> str:
        """Build the final prompt from the base instruction, context, and user query."""
        # TODO: YOUR CODE HERE
        if len(contexts) == 0:
            context = "1. [NO CONTEXT]"
        else:
            lines = []
            for i, ctx in enumerate(contexts, start=1):
                lines.append(f"{i}. {ctx}")
                context = "\n".join(lines)

        user_prompt = (
            self.base_system_prompt
            + "\n\nCONTEXT:\n"
            + context
            + "\n\nQUESTION:\n"
            + query
            + "\n\nAnswer:"
)


        #raise NotImplementedError
        return user_prompt

    def generate_answer(self, query: str) -> GenerationResult:
        """Full RAG pipeline: retrieve, augment, and answer."""
        # TODO: YOUR CODE HERE
        docs, scores = self.db.query(
        prompt=query,
        threshold=self.threshold,
        top_n=self.top_n,
        top_k=self.top_n,
        use_rerank=self.use_rerank,
    )

        if docs is None or scores is None or len(docs) == 0:
            augmented_prompt = self.augment(query, [])
            return GenerationResult(
                answer=self.fallback_answer,
                contexts=[],
                scores=[],
                augmented_prompt=augmented_prompt,
            )

        contexts = [str(x) for x in docs.tolist()]
        score_list = [float(x) for x in scores.tolist()]
        augmented_prompt = self.augment(query, contexts)

        query_tokens = set(query.lower().split())
        best_context = contexts[0]
        best_overlap = -1
        for ctx in contexts:
            overlap = len(query_tokens.intersection(ctx.lower().split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_context = ctx

        answer = best_context
        result = GenerationResult(
        answer=answer,
        contexts=contexts,
        scores=score_list,
        augmented_prompt=augmented_prompt,
    )
        #raise NotImplementedError
        return result


def build_pipeline(
    root_dir: Path | None = None,
    embedding_model_name: str | None = None,
    generation_model_name: str | None = None,
) -> tuple[VectorDB, RAGGenerator]:
    """Construct and initialize the pipeline."""

    config = build_reference_config(root_dir=root_dir)
    effective_generation_model_name = generation_model_name or config.generation.model_name
    seed_everything(config.seed, include_torch=(effective_generation_model_name is not None))
    ensure_directories(config.paths)

    if not hasattr(embedding_module, "transformer_cls"):
        from sentence_transformers import SentenceTransformer

        embedding_module.transformer_cls = SentenceTransformer

    embedder = SentenceEmbedder(
        model_name=embedding_model_name or config.embedding.model_name,
    )
    db = VectorDB(
        path2data=config.paths.data_dir,
        loaders={".pdf": PyMuPDFLoader},
        chunk_length=config.chunking.chunk_length,
        chunk_overlap=config.chunking.chunk_overlap,
        embedder=embedder,
        db_path=config.paths.db_dir,
        collection_name=config.paths.collection_name,
    )
    db.fill_db(reset_collection=True)

    generator = RAGGenerator(
        db=db,
        model_name=effective_generation_model_name,
        threshold=config.retrieval.threshold,
        top_n=config.retrieval.top_n,
        max_new_tokens=config.generation.max_new_tokens,
        fallback_answer=config.generation.fallback_answer,
        base_system_prompt=config.generation.base_system_prompt,
    )
    return db, generator


def load_demo_answers(
    questions: tuple[str, ...] | list[str] | None = None,
    output_path: Path | None = None,
) -> list[GenerationResult]:
    """Load answers for the questions from answers.json."""

    selected_questions = questions or DEFAULT_QUESTIONS
    saved_answers = load_answers(output_path=output_path)
    return [
        GenerationResult(
            answer=saved_answers[question],
            contexts=[],
            scores=[],
            augmented_prompt="",
        )
        for question in selected_questions
    ]


def write_demo_answers(
    questions: tuple[str, ...] | list[str] | None = None,
    root_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Generate answers and save them to answers.json."""

    _, generator = build_pipeline(root_dir=root_dir)
    selected_questions = questions or DEFAULT_QUESTIONS
    answers = [generator.generate_answer(question).answer for question in selected_questions]
    return save_answers(
        selected_questions,
        answers,
        output_path=output_path or default_answers_path(),
    )


def main() -> None:
    output_path = write_demo_answers()
    results = load_demo_answers(output_path=output_path)
    print(f"Saved answers to: {output_path}")
    for question, result in zip(DEFAULT_QUESTIONS, results, strict=True):
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print()


if __name__ == "__main__":
    main()
