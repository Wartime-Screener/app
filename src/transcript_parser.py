"""
Earnings call transcript parser.

Extracts key management commentary from raw earnings call transcripts
using keyword-based sentence extraction. Groups findings into categories
relevant to fundamental analysis.

No external LLM required — pure text parsing.
"""

import re
import logging

logger = logging.getLogger(__name__)


# Keywords for each category — order matters (first match wins for dedup)
CATEGORIES = {
    "Pricing & Revenue": [
        "price", "pricing", "revenue", "realization", "selling price",
        "average price", "per ton", "per barrel", "per gallon",
        "price environment", "pricing power", "rate",
    ],
    "Margins & Profitability": [
        "margin", "gross margin", "operating margin", "net margin",
        "ebitda", "profitab", "bottom line", "earnings per share",
        "eps", "net income", "operating income",
    ],
    "Costs & Efficiency": [
        "cost", "expense", "input cost", "raw material", "energy cost",
        "gas cost", "feedstock", "inflation", "efficiency", "SG&A",
        "cost reduction", "savings",
    ],
    "Demand & Volume": [
        "demand", "volume", "shipment", "ton", "production",
        "capacity", "utilization", "throughput", "output",
        "sales volume", "order", "backlog",
    ],
    "Capital & Investment": [
        "capex", "capital expenditure", "capital allocation",
        "investment", "project", "expansion", "construction",
        "acquisition", "buyback", "repurchase", "dividend",
        "return to shareholders", "share repurchase",
    ],
    "Outlook & Guidance": [
        "outlook", "guidance", "expect", "forecast", "anticipate",
        "going forward", "looking ahead", "next quarter", "next year",
        "full year", "fiscal year", "target", "goal", "plan to",
    ],
    "Risks & Headwinds": [
        "risk", "headwind", "challenge", "concern", "uncertainty",
        "tariff", "geopolit", "weather", "regulatory", "downturn",
        "disruption", "volatil", "inflationary pressure",
    ],
}

# Titles/roles that indicate management (not analysts)
MANAGEMENT_TITLES = [
    "ceo", "cfo", "coo", "president", "chief executive",
    "chief financial", "chief operating", "chairman", "vice president",
    "executive vice president", "evp", "svp", "treasurer", "secretary",
    "general counsel", "head of", "director of",
]


def _split_into_speaker_blocks(content: str) -> list[dict]:
    """
    Split transcript into speaker blocks.

    Returns list of {"speaker": str, "text": str} dicts.
    """
    # Match patterns like "John Smith:" or "Tony Will:" at start of line
    pattern = r'^([A-Z][a-zA-Z\s\.\-\']+?):\s*'
    parts = re.split(pattern, content, flags=re.MULTILINE)

    blocks = []
    # parts alternates: [preamble, speaker1, text1, speaker2, text2, ...]
    for i in range(1, len(parts) - 1, 2):
        speaker = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if speaker and text and len(speaker) < 60:
            blocks.append({"speaker": speaker, "text": text})

    return blocks


def _find_qa_boundary(content: str) -> int:
    """
    Find where the Q&A section begins.
    Returns character index, or len(content) if not found.
    """
    # Q&A-specific markers — must indicate transition TO questions
    markers = [
        "we will now take questions",
        "open the line for questions",
        "open it up for questions",
        "begin the question-and-answer",
        "begin our Q&A",
        "start the Q&A",
        "open the floor for questions",
        "take your questions",
        "our first question comes from",
        "our first question today",
        "That concludes our prepared remarks",
        "concludes our prepared comments",
        "that wraps up our prepared remarks",
    ]
    content_lower = content.lower()
    best_idx = len(content)
    for marker in markers:
        idx = content_lower.find(marker.lower())
        # Only accept if it's at least 10% into the transcript
        # (avoids matching intro boilerplate)
        if 0 < idx < best_idx and idx > len(content) * 0.10:
            best_idx = idx
    return best_idx


def _is_management_speaker(speaker: str, content: str) -> bool:
    """Check if a speaker is management (not an analyst or operator)."""
    speaker_lower = speaker.lower()
    if "operator" in speaker_lower:
        return False

    # Check if their title appears near their name in the transcript
    # (usually in the intro section)
    intro = content[:3000].lower()
    speaker_parts = speaker_lower.split()
    last_name = speaker_parts[-1] if speaker_parts else ""

    for title in MANAGEMENT_TITLES:
        # Check if the last name appears near a management title
        if last_name in intro:
            name_idx = intro.find(last_name)
            nearby = intro[max(0, name_idx - 100):name_idx + 100]
            if title in nearby:
                return True

    return False


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out short/noisy ones."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        # Filter: too short, too long, or just boilerplate
        if len(s) < 40 or len(s) > 600:
            continue
        # Skip legal disclaimers and boilerplate
        if any(skip in s.lower() for skip in [
            "forward-looking", "safe harbor", "not guarantees",
            "securities and exchange", "form 10-k", "form 10-q",
            "please note this event", "is being recorded",
            "good morning", "good afternoon", "good evening",
            "thanks for joining", "thank you for joining",
            "welcome to the", "with me today",
            "on this call, we'll", "discuss our outlook and then",
            "review the results",
        ]):
            continue
        result.append(s)
    return result


def parse_transcript(content: str, max_per_category: int = 4) -> dict:
    """
    Parse an earnings call transcript and extract key management commentary.

    Args:
        content: Raw transcript text.
        max_per_category: Maximum quotes per category.

    Returns:
        Dict with:
          - "categories": {category_name: [list of quote strings]}
          - "speakers": list of management speaker names found
          - "has_qa": bool indicating if Q&A section was detected
          - "prepared_remarks_pct": float, % of transcript that is prepared remarks
    """
    if not content:
        return {"categories": {}, "speakers": [], "has_qa": False,
                "prepared_remarks_pct": 0}

    # Find Q&A boundary
    qa_idx = _find_qa_boundary(content)
    has_qa = qa_idx < len(content)
    prepared_pct = (qa_idx / len(content) * 100) if len(content) > 0 else 0

    # Get speaker blocks
    blocks = _split_into_speaker_blocks(content)

    # Identify management speakers
    mgmt_speakers = set()
    for block in blocks:
        if _is_management_speaker(block["speaker"], content):
            mgmt_speakers.add(block["speaker"])

    # Extract sentences from prepared remarks by management
    prepared_blocks = _split_into_speaker_blocks(content[:qa_idx])
    mgmt_sentences = []
    for block in prepared_blocks:
        if block["speaker"] in mgmt_speakers or _is_management_speaker(block["speaker"], content):
            mgmt_sentences.extend(_extract_sentences(block["text"]))

    # If we didn't find management speakers reliably, fall back to
    # all prepared remarks (excluding Operator)
    if not mgmt_sentences:
        for block in prepared_blocks:
            if "operator" not in block["speaker"].lower():
                mgmt_sentences.extend(_extract_sentences(block["text"]))

    # Also extract key Q&A responses from management
    if has_qa:
        qa_blocks = _split_into_speaker_blocks(content[qa_idx:])
        for block in qa_blocks:
            if block["speaker"] in mgmt_speakers or _is_management_speaker(block["speaker"], content):
                mgmt_sentences.extend(_extract_sentences(block["text"]))

    # Categorize sentences
    categories = {}
    used_sentences = set()

    for cat_name, keywords in CATEGORIES.items():
        matches = []
        for sent in mgmt_sentences:
            if id(sent) in used_sentences:
                continue
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                matches.append(sent)
                used_sentences.add(id(sent))
                if len(matches) >= max_per_category:
                    break

        if matches:
            categories[cat_name] = matches

    return {
        "categories": categories,
        "speakers": sorted(mgmt_speakers),
        "has_qa": has_qa,
        "prepared_remarks_pct": round(prepared_pct, 1),
    }
