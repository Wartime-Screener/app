"""
LLM-powered earnings call transcript summarizer.

Uses Claude (Anthropic API) to extract and categorize key management
commentary from earnings call transcripts. Much higher quality than
keyword-based extraction — understands context, nuance, and intent.

Uses Claude Haiku for cost efficiency (~$0.01 per transcript).
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "transcripts"

# Categories for management commentary extraction
CATEGORIES = [
    "Pricing & Revenue",
    "Margins & Profitability",
    "Costs & Efficiency",
    "Demand & Volume",
    "Capital & Investment",
    "Outlook & Guidance",
    "Risks & Headwinds",
]

SYSTEM_PROMPT = """You are a financial analyst assistant. Your job is to read earnings call transcripts and extract the most important management commentary, organized into categories.

Focus ONLY on what management (CEO, CFO, executives) said — ignore analyst questions and operator statements.

For each category, extract 2-4 of the most insightful, specific quotes or paraphrased points. Prefer:
- Concrete numbers and figures over vague statements
- Forward-looking commentary over backward-looking recaps
- Specific product/segment commentary over generic platitudes
- Anything surprising, unusual, or that would move a stock price

Skip any category where management didn't say anything meaningful.

Return your response as valid JSON with this exact structure:
{
  "categories": {
    "Category Name": ["point 1", "point 2", "point 3"],
    ...
  },
  "summary": "2-3 sentence executive summary of the most important takeaways from this call",
  "sentiment": "bullish" | "neutral" | "bearish",
  "key_numbers": ["list of the most important specific numbers/metrics mentioned"]
}

Categories to use (only include ones with meaningful content):
- Pricing & Revenue
- Margins & Profitability
- Costs & Efficiency
- Demand & Volume
- Capital & Investment
- Outlook & Guidance
- Risks & Headwinds

IMPORTANT: Return ONLY the JSON object. No markdown, no code fences, no explanation."""


def _cache_key(ticker: str, year: int, quarter: int) -> str:
    raw = f"{ticker}|{year}|Q{quarter}"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_cache(cache_key: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        cached_at = datetime.fromisoformat(data["cached_at"])
        # Cache transcript summaries for 90 days (they never change once published)
        if datetime.now() - cached_at < timedelta(days=90):
            return data["response"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _write_cache(cache_key: str, response_data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    payload = {
        "cached_at": datetime.now().isoformat(),
        "response": response_data,
    }
    try:
        cache_path.write_text(json.dumps(payload, default=str))
    except Exception as e:
        logger.warning(f"Failed to write transcript cache: {e}")


def summarize_transcript(
    content: str,
    ticker: str = "",
    year: int = 0,
    quarter: int = 0,
    skip_cache: bool = False,
) -> dict:
    """
    Summarize an earnings call transcript using Claude.

    Args:
        content: Raw transcript text.
        ticker: Stock ticker (for caching).
        year: Earnings year (for caching).
        quarter: Earnings quarter (for caching).

    Returns:
        Dict with:
          - "categories": {category_name: [list of commentary strings]}
          - "summary": executive summary string
          - "sentiment": "bullish" | "neutral" | "bearish"
          - "key_numbers": list of key metrics mentioned
          - "source": "claude" (to distinguish from keyword parser)
    """
    if not content:
        return {
            "categories": {},
            "summary": "",
            "sentiment": "neutral",
            "key_numbers": [],
            "source": "claude",
        }

    # Check cache first
    if ticker and year and quarter and not skip_cache:
        ck = _cache_key(ticker, year, quarter)
        cached = _read_cache(ck)
        if cached is not None:
            logger.info(f"Using cached transcript summary for {ticker} Q{quarter} {year}")
            return cached

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set.")
        return {
            "categories": {},
            "summary": "Anthropic API key not configured.",
            "sentiment": "neutral",
            "key_numbers": [],
            "source": "error",
        }

    # Truncate very long transcripts to stay within context window
    # Haiku handles 200K tokens, but let's keep it reasonable for cost
    max_chars = 100_000  # ~25K tokens
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Transcript truncated for length]"

    try:
        client = anthropic.Anthropic(api_key=api_key)

        user_msg = f"Here is the {ticker} Q{quarter} {year} earnings call transcript. Extract and categorize the key management commentary.\n\n---\n\n{content}"

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        # Handle potential markdown code fences
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        parsed = json.loads(response_text)

        result = {
            "categories": parsed.get("categories", {}),
            "summary": parsed.get("summary", ""),
            "sentiment": parsed.get("sentiment", "neutral"),
            "key_numbers": parsed.get("key_numbers", []),
            "source": "claude",
        }

        # Cache the result
        if ticker and year and quarter:
            _write_cache(ck, result)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        # Return the raw text as a single summary if JSON parsing fails
        return {
            "categories": {},
            "summary": response_text[:500] if 'response_text' in dir() else "Failed to parse response.",
            "sentiment": "neutral",
            "key_numbers": [],
            "source": "claude",
        }
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return {
            "categories": {},
            "summary": f"API error: {str(e)[:200]}",
            "sentiment": "neutral",
            "key_numbers": [],
            "source": "error",
        }
    except Exception as e:
        logger.error(f"Transcript summarization failed: {e}")
        return {
            "categories": {},
            "summary": f"Error: {str(e)[:200]}",
            "sentiment": "neutral",
            "key_numbers": [],
            "source": "error",
        }


def is_configured() -> bool:
    """Check if the Anthropic API key is set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    return key is not None and len(key) > 0
