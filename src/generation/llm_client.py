# src/generation/llm_client.py
"""
LLM Client — calls Claude API and returns a grounded, cited answer.

Model: claude-sonnet-4-6 (set in config.py)

Why Anthropic Claude for RAG?
    Claude excels at instruction-following and citation compliance.
    In RAG systems, the LLM must stay strictly within the provided context
    and cite every claim. Claude reliably does this with a well-structured
    prompt — hallucination rates are lower than most alternatives when
    given explicit "cite everything" instructions.

Temperature = 0.1:
    Low temperature = more deterministic, more factual.
    High temperature = more creative, more unpredictable.
    For a document Q&A system, unpredictability is a bug, not a feature.
    0.1 is the standard for production RAG systems.

Langfuse observability:
    Every API call is traced if LANGFUSE_PUBLIC_KEY is set in .env.
    Traces include: prompt, response, token count, latency, model name.
    If Langfuse keys are not set, the app works normally without tracing.
"""

from dataclasses import dataclass
from typing import Optional

import anthropic
from loguru import logger

from config import get_settings
from src.generation.context_builder import BuiltContext


# ── Data Structure ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    """
    The complete output of one LLM call.

    Attributes:
        answer:          Claude's response text with [Source N] citations.
        input_tokens:    Tokens consumed by the prompt.
        output_tokens:   Tokens consumed by the response.
        model:           Model name used (from config).
        context:         The BuiltContext that produced this answer.
                         Stored for hallucination scoring in Stage 7.
    """
    answer:        str
    input_tokens:  int
    output_tokens: int
    model:         str
    context:       BuiltContext

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_estimate_usd(self) -> float:
        """
        Rough cost estimate based on Sonnet pricing.
        Input: $3/M tokens, Output: $15/M tokens (approximate).
        """
        input_cost  = (self.input_tokens  / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return round(input_cost + output_cost, 6)


# ── LLM Client ─────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Sends a BuiltContext to Claude and returns a GeneratedAnswer.

    Usage:
        client = LLMClient()
        answer = client.generate(context)
        print(answer.answer)         # Claude's response
        print(answer.total_tokens)   # tokens used
    """

    def __init__(self):
        self.settings = get_settings()

        # Anthropic client reads ANTHROPIC_API_KEY from env automatically
        # when no api_key argument is passed.
        self.client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key,
        )

        # Initialise Langfuse if keys are configured
        self._langfuse = self._init_langfuse()

        logger.info(
            "LLMClient ready | model='{}' | temperature={} | "
            "observability={}",
            self.settings.llm_model,
            self.settings.temperature,
            "enabled" if self._langfuse else "disabled",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        context: BuiltContext,
        trace_name: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        Send context to Claude and return the generated answer.

        Args:
            context:    BuiltContext from ContextBuilder.build()
            trace_name: Optional name for the Langfuse trace
                        (e.g. the user's question for easy identification)

        Returns:
            GeneratedAnswer with answer text, token counts, and cost estimate.

        Raises:
            anthropic.APIError: If the API call fails.
        """
        logger.info(
            "Calling Claude '{}' | ~{} prompt tokens",
            self.settings.llm_model,
            context.total_tokens_est,
        )

        # Start Langfuse trace if available
        trace = self._start_trace(trace_name or "rag_query", context)

        try:
            response = self.client.messages.create(
                model      = self.settings.llm_model,
                max_tokens = self.settings.max_tokens,
                temperature= self.settings.temperature,
                system     = context.system_prompt,
                messages   = [
                    {"role": "user", "content": context.user_prompt}
                ],
            )

            answer_text   = response.content[0].text
            input_tokens  = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            result = GeneratedAnswer(
                answer        = answer_text,
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
                model         = self.settings.llm_model,
                context       = context,
            )

            # Log to Langfuse
            self._end_trace(trace, result)

            logger.success(
                "Claude responded | {} input + {} output tokens | "
                "est. cost=${:.5f}",
                input_tokens,
                output_tokens,
                result.cost_estimate_usd,
            )

            return result

        except anthropic.APIError as e:
            logger.error("Claude API error: {}", e)
            if trace:
                self._end_trace(trace, None, error=str(e))
            raise

    # ── Private: Langfuse ──────────────────────────────────────────────────────

    def _init_langfuse(self):
        """
        Initialise Langfuse client if keys are configured.
        Returns None if Langfuse is not configured — app works either way.
        """
        if not self.settings.langfuse_public_key:
            logger.debug("Langfuse not configured — observability disabled")
            return None

        try:
            from langfuse import Langfuse
            lf = Langfuse(
                public_key  = self.settings.langfuse_public_key,
                secret_key  = self.settings.langfuse_secret_key,
                host        = self.settings.langfuse_host,
            )
            logger.info("Langfuse observability enabled")
            return lf
        except Exception as e:
            logger.warning("Langfuse init failed (non-fatal): {}", e)
            return None

    def _start_trace(self, name: str, context: BuiltContext):
        """Start a Langfuse trace. Returns None if Langfuse is disabled."""
        if not self._langfuse:
            return None
        try:
            return self._langfuse.trace(
                name    = name,
                input   = context.user_prompt[:500],  # truncate for readability
                metadata= {
                    "model":             self.settings.llm_model,
                    "temperature":       self.settings.temperature,
                    "estimated_tokens":  context.total_tokens_est,
                    "num_sources":       len(context.source_map),
                    "num_images":        len(context.image_map),
                },
            )
        except Exception as e:
            logger.debug("Langfuse trace start failed (non-fatal): {}", e)
            return None

    def _end_trace(
        self,
        trace,
        result: Optional[GeneratedAnswer],
        error: Optional[str] = None,
    ):
        """Complete a Langfuse trace with output and token usage."""
        if not trace:
            return
        try:
            if result:
                trace.update(
                    output   = result.answer[:500],
                    metadata = {
                        "input_tokens":   result.input_tokens,
                        "output_tokens":  result.output_tokens,
                        "cost_usd":       result.cost_estimate_usd,
                    },
                )
            elif error:
                trace.update(
                    output   = f"ERROR: {error}",
                    level    = "ERROR",
                )
        except Exception as e:
            logger.debug("Langfuse trace end failed (non-fatal): {}", e)