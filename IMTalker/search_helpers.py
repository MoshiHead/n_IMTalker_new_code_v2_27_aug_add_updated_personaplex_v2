"""search_helpers.py — query routing, web search, context compression, and the
isolated upstream-`moshi` STT loader for the live IMTalker avatar pipeline
(`liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py`).

This replaces the former `rag_helpers.py`. The local-document RAG path (FAISS
index, sentence-transformers embeddings, hierarchical chunk retrieval) has been
removed entirely. The pipeline is now:

    user speech -> STT transcript -> QueryRouter decision
        |                                   |
        |                        needs_search == False
        |                                   |
        |                          model answers from its
        |                          own knowledge (no injection)
        |
        +-- needs_search == True -> web search -> clean -> compress
                                      -> inject <ref> ... <ref>

Everything here is intentionally free of any dependency on the PersonaPlex
`moshi` fork, `lm_gen`, or CUDA-graph state — these are pure decision/
retrieval/compression/HTTP helpers plus one namespaced import shim, safe to
call from a background thread. The only GPU-touching classes are
`ContextCompressor` and `QueryRouter`, which share one small model and never
touch the main PersonaPlex LM.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_SYMBOL_RE = re.compile(r"[*_#`~]+")


# ── Text-injection tag helpers ──────────────────────────────────────────────

def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def wrap_with_ref_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<ref>") and cleaned.endswith("<ref>"):
        return cleaned
    return f"<ref> {cleaned} <ref>"


def wrap_with_lookup_tags() -> str:
    return "<lookup> Please wait a minute."


# ── STT transcript decoding ──────────────────────────────────────────────────

def decode_stt_tokens(text_tokens: list[torch.Tensor], tokenizer, padding_token_id: int) -> str:
    if not text_tokens:
        return ""
    all_tokens = torch.cat(text_tokens, dim=-1)
    all_tokens = all_tokens.cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return ""
    return tokenizer.decode(valid.tolist())


def decode_stt_tokens_with_ids(
    text_tokens: list[torch.Tensor], tokenizer, padding_token_id: int
) -> tuple[str, list[int]]:
    """Same as decode_stt_tokens, but also returns the surviving token ids.

    The ids are what make a bad transcript diagnosable. If the decoded text is
    in a script the model cannot produce, the ids tell you immediately whether
    the model emitted something strange or whether the ids are fine and the
    *tokenizer* decoding them is the wrong one (see `describe_tokenizer`)."""
    if not text_tokens:
        return "", []
    all_tokens = torch.cat(text_tokens, dim=-1).cpu().view(-1)
    valid = all_tokens[all_tokens > padding_token_id]
    if valid.numel() == 0:
        return "", []
    ids = [int(i) for i in valid.tolist()]
    return tokenizer.decode(ids), ids


# Injected control tags echoed back by the model. The <lookup>/<ref> blocks are
# fed in as ordinary text tokens, and the model sometimes reproduces them in its
# own output stream -- confirmed in conversation_log_1 turn 4, whose recorded
# reply was
#   ".> <ref> Today's Gold market price is $4,040.91. It's not provided ... <ref>"
# i.e. the tag markup and a stray fragment of the preceding injection ended up
# inside the assistant's transcript. Stripping them keeps the recorded reply
# readable and stops tag text being mistaken for something the user heard.
#
# NOTE this is presentation only. It cannot stop the model from *speaking* an
# echoed tag; that is a model/adapter behavior, not something the transcript
# layer can reach.
_ECHOED_TAG_RE = re.compile(r"<\s*/?\s*(?:ref|lookup|system)\s*>", re.IGNORECASE)
_ORPHAN_TAG_TAIL_RE = re.compile(r"^[\s.>]*>")


def strip_injected_tags(text: str) -> str:
    """Remove echoed <ref>/<lookup>/<system> markup and the leading '>'
    fragment left when a turn boundary lands mid-tag."""
    if not text:
        return ""
    cleaned = _ECHOED_TAG_RE.sub(" ", text)
    cleaned = _ORPHAN_TAG_TAIL_RE.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def describe_tokenizer(tokenizer) -> str:
    """Best-effort identity of a tokenizer, for startup logging.

    Vocabulary size is the cheap tell for a tokenizer mix-up: the Kyutai
    en/fr STT tokenizer and PersonaPlex's 32k multilingual SPM are very
    different sizes, so a wrong-tokenizer bug shows up here as a wrong number
    long before it shows up as a strange transcript."""
    for attr in ("get_piece_size", "vocab_size", "__len__"):
        try:
            v = getattr(tokenizer, attr)
            n = int(v() if callable(v) else v)
            return f"{type(tokenizer).__name__}(vocab_size={n})"
        except Exception:
            continue
    return type(tokenizer).__name__


# ── Transcript sanity: script/language guard ────────────────────────────────
#
# The STT submodel this pipeline ships with (kyutai/stt-1b-en_fr) only produces
# English and French. Any transcript containing a meaningful amount of
# non-Latin script is therefore not a translation or a language surprise -- it
# is decode garbage, and it must never reach the router or a web search. A
# garbage transcript that reaches the router is worse than no transcript: it
# routes a question nobody asked, and can send nonsense to a paid search API.

def _is_latin_letter(ch: str) -> bool:
    """True for a-z/A-Z and accented Latin (French needs e-acute, c-cedilla,
    ...). Uses the Unicode character NAME rather than a codepoint range so
    accented Latin is kept and Devanagari/Cyrillic/CJK/Arabic are not."""
    if not ch.isalpha():
        return False
    if ch.isascii():
        return True
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def transcript_script_stats(text: str) -> dict:
    """Letter-only script breakdown. Digits, spaces and punctuation are
    ignored: they are script-neutral and would dilute the ratio."""
    letters = [c for c in (text or "") if c.isalpha()]
    non_latin = [c for c in letters if not _is_latin_letter(c)]
    total = len(letters)
    return {
        "letters": total,
        "non_latin": len(non_latin),
        "non_latin_ratio": (len(non_latin) / total) if total else 0.0,
        "sample": "".join(dict.fromkeys(non_latin))[:24],
    }


def check_transcript_script(text: str, max_non_latin_ratio: float = 0.15) -> tuple[bool, dict]:
    """Returns (looks_usable, stats).

    `max_non_latin_ratio` defaults to 0.15 rather than 0: a stray decoded
    symbol in an otherwise clean English sentence should not throw the turn
    away, but a transcript that is mostly another script must be rejected."""
    stats = transcript_script_stats(text)
    if stats["letters"] == 0:
        return False, stats
    return stats["non_latin_ratio"] <= float(max_non_latin_ratio), stats


# ── English-only guard ──────────────────────────────────────────────────────
#
# The script check above only catches other WRITING SYSTEMS. The bundled STT
# model is bilingual (en/fr) and, on unclear audio, will happily hallucinate
# fluent Spanish or French in ordinary Latin letters -- confirmed in
# conversation_log_1: "Bien nino, tu eres mi How Bitcoin World." and
# "Allez, Helmut." both passed the script check because they are Latin.
#
# Detection is deliberately lexical rather than model-based: it must be able to
# run inline on the GPU thread inside _step()'s 80ms budget, so a language-ID
# model is not an option. Function words are the right signal because they are
# the highest-frequency tokens in any utterance and are strongly language-
# specific, unlike content words (brand names, numbers) which are shared.

_ENGLISH_FUNCTION_WORDS = frozenset("""
a an the is are was were be been being am do does did doing done have has had
i me my mine we us our you your he him his she her it its they them their this
that these those what which who whom whose when where why how there here
and or but if then than because so as of to in on at by for from with without
about into over under again once all any both each few more most other some such
no nor not only own same too very can could will would shall should may might must
please thanks thank hello hey hi okay ok yes yeah yep nope tell say said know
think want need get got give take make made go going come came see look
today tomorrow yesterday now price stock market news weather time
""".split())

# Words that are strong evidence of Spanish / French / German / Italian /
# Portuguese AND are not themselves English words. Anything ambiguous is
# excluded on purpose -- "no", "was", "die", "come", "con", "per", "pour",
# "comment", "plus", "hay", "dove" are all real English words and would cause
# false rejections of perfectly good English.
_FOREIGN_FUNCTION_WORDS = frozenset("""
que eres muy pero para como hola gracias nino senor esta estoy soy bien
los las una por del ese esa tiene hacer puede donde cuando ustedes senora
je vous nous elle ils elles les des une est sont etre avoir avec dans mais
qui quoi pourquoi bonjour merci oui allez allons tres cette ces tout tous
faire peut sur alors donc aussi voila jour
der das ich sie wir ihr und oder aber nicht sind haben sein mit fuer auf
wie wann warum danke hallo nein eine einen ist ja nein kein
che non sono gli quando perche ciao grazie questo questa
nao sim uma mas onde porque ola obrigado voce muito
""".split())

# Accented forms carry the same weight; normalizing to ASCII lets one list
# cover both "nino"/"nino" and "tres"/"tres" as they may arrive either way.
_FOREIGN_FUNCTION_WORDS = _FOREIGN_FUNCTION_WORDS - _ENGLISH_FUNCTION_WORDS

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip_accents(word: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn"
    )


def transcript_language_stats(text: str) -> dict:
    """Lexical evidence for/against the transcript being English."""
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    ascii_words = [_strip_accents(w) for w in words]
    english_hits = sum(1 for w in ascii_words if w in _ENGLISH_FUNCTION_WORDS)
    foreign = [w for w in ascii_words if w in _FOREIGN_FUNCTION_WORDS]
    letters = [c for c in (text or "") if c.isalpha()]
    # English essentially never uses diacritics; French and Spanish use them
    # constantly. A high rate here is independent corroboration.
    accented = [c for c in letters if _is_latin_letter(c) and not c.isascii()]
    return {
        "words": len(words),
        "english_hits": english_hits,
        "foreign_hits": len(foreign),
        "foreign_words": sorted(set(foreign))[:8],
        "accented_ratio": (len(accented) / len(letters)) if letters else 0.0,
    }


def check_transcript_english(text: str) -> tuple[bool, dict]:
    """Returns (looks_english, stats).

    Conservative by construction -- it demands positive evidence of another
    language and defaults to accepting. Rejecting a valid English question is
    worse than passing one foreign word through, because a rejected turn gets
    no answer at all.

    Rejects when any of:
      * two or more unambiguous foreign function words
      * one foreign function word and NO English function words (covers short
        utterances like "Allez, Helmut." where there is nothing English to
        weigh against it)
      * one foreign function word plus a meaningful rate of accents
    """
    stats = transcript_language_stats(text)
    f, e = stats["foreign_hits"], stats["english_hits"]
    if f >= 2:
        stats["reason"] = f"{f} non-English function words {stats['foreign_words']}"
        return False, stats
    if f >= 1 and e == 0:
        stats["reason"] = (
            f"non-English word {stats['foreign_words']} with no English function words"
        )
        return False, stats
    if f >= 1 and stats["accented_ratio"] > 0.05:
        stats["reason"] = (
            f"non-English word {stats['foreign_words']} plus "
            f"{stats['accented_ratio']:.0%} accented letters"
        )
        return False, stats
    stats["reason"] = ""
    return True, stats


def check_transcript_usable(
    text: str, max_non_latin_ratio: float = 0.15, require_english: bool = True,
) -> tuple[bool, dict]:
    """Single entry point: writing system first, then language. Returns
    (usable, stats) with `stats['reason']` explaining any rejection."""
    ok_script, stats = check_transcript_script(text, max_non_latin_ratio)
    if not ok_script:
        stats["reason"] = (
            "empty transcript" if stats["letters"] == 0
            else f"{stats['non_latin_ratio']:.0%} non-Latin script (sample {stats['sample']!r})"
        )
        stats["kind"] = "script"
        return False, stats
    if not require_english:
        stats["reason"] = ""
        stats["kind"] = ""
        return True, stats
    ok_lang, lang_stats = check_transcript_english(text)
    stats.update(lang_stats)
    stats["kind"] = "" if ok_lang else "language"
    return ok_lang, stats


# ── Tier 0: instant rule-based routing ──────────────────────────────────────
#
# Four ordered layers, pure regex, microseconds. They run inline on the GPU
# thread so the common shapes never pay for a model forward pass, and -- more
# importantly -- so the common shapes are decided DETERMINISTICALLY instead of
# being left to a 1.5B model's judgement.
#
# The ordering is the whole design. Earlier layers win:
#
#   0 SELF            the question is about the ASSISTANT ("what's your name",
#                     "how are you", "can you hear me") -> NO SEARCH. Above the
#                     time layer on purpose: "how are you today?" carries a time
#                     word and is still small talk.
#   0 GREETING_ONLY   the whole utterance is a greeting or a thank-you, with no
#                     question attached -> NO SEARCH. Also above the time layer,
#                     but only when nothing else was said: "good morning" is
#                     small talk, "good morning, what's the weather today?" is
#                     still a weather question.
#   1 EXPLICIT_TIME   an explicit time reference ("today", "right now", "2026")
#                     -> SEARCH. A time word is the strongest possible signal
#                     and must beat everything below it, so "what is the gold
#                     price today" searches even though "what is" looks static.
#   2 STATIC_FORM     the question's SHAPE is conceptual, procedural or
#                     advisory -> NO SEARCH, whatever the topic.
#   3 LIVE_TOPIC      asks for a value or event that is inherently live
#                     (a price, the weather, a score) -> SEARCH.
#   4 CHITCHAT        greetings, thanks, references to the conversation itself
#                     -> NO SEARCH.
#
# Layer 2 sits ABOVE layer 3 on purpose, and that is the fix for the misroutes
# in conversation_log_2. "How I can invest in cryptocurrency?" and "Is it safe
# to invest in cryptocurrency?" are finance questions -- topic words like
# invest/crypto/market pull hard towards "live data" -- but their SHAPE is
# procedural and advisory, and shape is what actually determines whether a
# lookup is needed. Judge the question type, not its subject.

_RE_FLAGS = re.IGNORECASE

# -- Layer 0: questions about the ASSISTANT itself ----------------------------
# The one category no search can ever help with: the answer lives in the
# persona/system prompt, not on the web. Sending "can you tell me your name?" to
# a search engine costs 2-4s of thinking sound and then grounds a friendly
# greeting in a stranger's web page.
#
# Before this layer existed these questions matched NO rule at all and were
# handed to the 1.5B router, whose threshold is deliberately biased toward
# searching (0.40) -- so the most common opening line in any conversation was
# decided by a coin flip. They are decided here instead, deterministically and
# with no model call.
#
# The patterns are tight on purpose: they require the assistant to be the
# SUBJECT of the question, not merely mentioned in it. "Can you tell me the gold
# price today?" also contains "can you", and must still search.
_RULE_SELF_RE = re.compile(
    # "what is your name", "what's your favourite colour"
    r"\byour\s+(?:name|age|job|role|purpose|creator|maker|owner|birthday|"
    r"favou?rite|hobby|hobbies|feelings?)\b"
    r"|\bwhat(?:'s|s|\s+is|\s+are|\s+was)\s+your\b"
    r"|\b(?:what|who|how)\s+(?:should|shall|do|can|may)\s+i\s+call\s+you\b"
    # identity: "who are you", "what are you", "who made you"
    r"|\bwho\s+(?:are|r)\s+(?:you|u)\b"
    r"|\bwhat\s+are\s+you\b"
    r"|\bwho\s+(?:made|created|built|trained|designed|programmed|developed)\s+you\b"
    r"|\bam\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)\b"
    # state / small talk aimed at the assistant
    r"|\bhow\s+(?:are|r)\s+(?:you|u)\b|\bhow'?re\s+you\b|\bhow\s+(?:you|u)\s+doing\b"
    r"|\bhow\s+have\s+you\s+been\b|\bhow\s+(?:is|was|'s)\s+your\s+day\b"
    r"|\bhow\s+old\s+are\s+you\b|\bwhere\s+are\s+you\s+from\b"
    # presence checks -- extremely common as the first utterance of a session
    r"|\bare\s+you\s+(?:a|an|the|there|real|human|alive|awake|listening|hearing|"
    r"okay|ok|fine|sure|ready|busy|free|still|able)\b"
    r"|\bcan\s+you\s+(?:hear|see|understand)\s+(?:me|us)\b"
    r"|\bcan\s+you\s+(?:hear|see)\b"
    # preferences and capabilities of the assistant. Note the verb list is
    # restricted to personal ones: "do you know" and "do you have" are NOT here,
    # because "do you know the weather today?" is a weather question.
    # The optional adverb matters: "do you EVER get tired" is the natural
    # phrasing and a strict "do you get tired" misses it entirely.
    r"|\bdo\s+you\s+(?:ever\s+|really\s+|actually\s+|even\s+)?"
    r"(?:like|love|enjoy|prefer|dream|sleep|eat|believe|remember\s+me|"
    r"get\s+tired|get\s+bored|have\s+feelings|have\s+a\s+name)\b"
    r"|\bwhat\s+(?:can|do)\s+you\s+do\b|\bwhat\s+else\s+can\s+you\s+do\b"
    # "tell me about yourself", "introduce yourself"
    r"|\byourself\b",
    _RE_FLAGS,
)

# -- Layer 0: pure greetings / sign-offs / acknowledgements --------------------
# Matched here only when the greeting is the WHOLE utterance (see
# _greeting_only_match). "Good morning" needs no search; "good morning, what's
# the weather today?" is still a weather question, and the leftover-content test
# is what keeps those apart.
_RULE_GREETING_RE = re.compile(
    r"\b("
    r"hello|hi|hey|yo|howdy|greetings|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"how(?:'s|s|\s+is)\s+it\s+going|how(?:'s|s|\s+is)\s+everything|"
    r"what(?:'s|s|\s+is)\s+up|sup|"
    r"thank\s+you|thanks|much\s+appreciated|appreciate\s+it|cheers|"
    r"good\s?bye|bye|see\s+you|see\s+ya|take\s+care|good\s+night|"
    r"have\s+a\s+(?:nice|good|great)\s+(?:day|one|evening|night)|"
    r"nice\s+to\s+meet\s+you|pleasure\s+to\s+meet\s+you|long\s+time\s+no\s+see|"
    r"you(?:'re|r)\s+welcome|no\s+problem|sounds\s+good|got\s+it|understood|"
    r"alright|okay|ok|cool|awesome|perfect|excellent"
    r")\b",
    _RE_FLAGS,
)

# Words that do not count as "something else was asked" when deciding whether an
# utterance is nothing but a greeting.
_GREETING_FILLER_RE = re.compile(
    r"\b(?:please|there|again|so|well|just|now|very|much|lot|lots|dear|sir|madam|"
    r"ma'am|man|buddy|friend|mate|guys|everyone|my|to|you|u|me|it|is|a|an|the|and|"
    r"then|too|also|for|that|this|of)\b",
    _RE_FLAGS,
)

_WORD_RE = re.compile(r"[A-Za-z']+")

# How many leftover words an utterance may still contain and count as "nothing
# but a greeting". 2 keeps "thanks a lot", "hi there friend" and "okay cool" in,
# while "hi, what is the weather today" (4 leftover words) drops through to the
# time layer below and still searches.
_GREETING_MAX_LEFTOVER_WORDS = 2


def _greeting_only_match(text: str) -> str:
    """The matched greeting when `text` is nothing BUT a greeting, else "".

    Deliberately not a plain regex match: the point is not whether a greeting
    appears, but whether anything else was asked alongside it."""
    m = _RULE_GREETING_RE.search(text)
    if not m:
        return ""
    rest = _RULE_GREETING_RE.sub(" ", text)
    rest = _GREETING_FILLER_RE.sub(" ", rest)
    if len(_WORD_RE.findall(rest)) <= _GREETING_MAX_LEFTOVER_WORDS:
        return m.group(0).strip()
    return ""


# -- Layer 1: an explicit reference to *when* ---------------------------------
_RULE_TIME_RE = re.compile(
    r"\b("
    r"today|tonight|right now|just now|at the moment|at present|"
    r"currently|current|latest|newest|most recent|recently|these days|"
    r"this (?:week|month|year|morning|afternoon|evening|quarter)|"
    r"yesterday|tomorrow|last night|so far (?:this|today)|"
    r"up to date|up-to-date|as of (?:now|today)|"
    r"20(?:2[5-9]|[3-9]\d)"
    r")\b",
    _RE_FLAGS,
)

# -- Layer 2: conceptual / procedural / advisory question shapes --------------
# Each entry is (label, pattern). The label is logged so a surprising verdict
# can be traced to the exact rule that produced it.
_RULE_STATIC_FORMS = [
    # Procedural: "how do I ...". Also accepts the scrambled word order real
    # STT produces -- conversation_log_2 transcribed the failing turn as
    # "How I can invest in cryptocurrency?", with "I" before "can", which a
    # strict `how can i` pattern would miss entirely.
    ("procedural", re.compile(
        r"\bhow\s+(?:do|can|could|would|should|might|may)\s+(?:i|we|you|someone|one|a person)\b"
        r"|\bhow\s+(?:i|we|you)\s+(?:can|could|do|would|should)\b"
        r"|\bhow\s+to\s+\w+"
        r"|\b(?:what|which)\s+(?:is|are)\s+the\s+(?:steps|process|procedure|best way|easiest way|"
        r"safest way|right way)\b"
        r"|\bsteps\s+to\s+\w+|\bguide\s+to\s+\w+|\bhow\s+does\s+(?:one|someone)\b",
        _RE_FLAGS)),
    # Mechanism / causation: "how does X work", "why is X ...".
    ("mechanism", re.compile(
        r"\bhow\s+(?:does|do|did|is|are|was|were)\b[^?]*\b"
        r"(?:work|works|working|determined|calculated|decided|set|made|created|produced|formed|"
        r"generated|priced)\b"
        r"|\bwhy\s+(?:does|do|did|is|are|was|were|would|should|can|would)\b"
        r"|\bhow come\b",
        _RE_FLAGS)),
    # Advisory / judgement: "is it safe to ...", "should I ...".
    ("advisory", re.compile(
        r"\bis\s+it\s+(?:safe|good|wise|smart|worth|legal|okay|ok|advisable|risky|bad|better|"
        r"possible|practical|realistic|sensible)\b"
        r"|\b(?:is|are)\s+\w[\w\s]{0,30}?\s+(?:safe|risky|legal|reliable|trustworthy|worth it|"
        r"a good idea|a bad idea|a safe investment|a good investment)\b"
        r"|\bshould\s+(?:i|we|you|one)\b|\bwhat\s+should\s+\w+\s+(?:consider|know|look at|watch)\b"
        r"|\b(?:do|would)\s+you\s+(?:recommend|suggest|advise)\b"
        r"|\bwhat\s+do\s+you\s+(?:think|recommend|suggest|advise)\b"
        r"|\bis\s+it\s+a\s+(?:good|bad|smart|wise)\b",
        _RE_FLAGS)),
    # Conceptual: differences, risks, benefits, definitions of a concept.
    ("conceptual", re.compile(
        r"\bwhat\s+(?:is|are)\s+the\s+(?:difference|differences|benefit|benefits|risk|risks|"
        r"advantage|advantages|disadvantage|disadvantages|downside|downsides|point|purpose|"
        r"meaning|pros|cons)\b"
        r"|\b(?:pros and cons|advantages and disadvantages)\b"
        r"|\b(?:explain|describe|define)\b"
        r"|\btell me about\b"
        r"|\bwhat does\b[^?]*\bmean\b"
        r"|\bwhat\s+(?:is|are)\s+\w+\s+used for\b",
        _RE_FLAGS)),
    # Historical: settled facts about the past. Note "when WAS" (past) only --
    # "when IS the next meeting" is a schedule and must stay searchable.
    ("historical", re.compile(
        r"\bwhen\s+(?:was|were|did)\b"
        r"|\bwho\s+(?:invented|created|founded|wrote|discovered|started|built|designed)\b"
        r"|\bhistory of\b|\borigin of\b|\bwhere does\b[^?]*\bcome from\b",
        _RE_FLAGS)),
]

# -- Layer 3: inherently live values and events -------------------------------
# Deliberately narrower than "finance words". Bare "stock" or "market" is NOT
# here: "what is a stock" and "how does the stock market work" are textbook
# explanations. Only phrasings that request a VALUE or an EVENT qualify.
_RULE_LIVE_TOPIC_RE = re.compile(
    r"\b("
    r"price|prices|pricing|quote|quotes|"
    r"stock price|share price|market cap|market value|exchange rate|conversion rate|"
    r"interest rate|mortgage rate|"
    r"weather|forecast|temperature (?:outside|today|now)|"
    r"news|headline|headlines|breaking|"
    r"score|scores|standings|fixtures|results of|who won|winner of|"
    r"trading at|worth right now|"
    r"who is the (?:current|new|president|ceo|prime minister|chairman)"
    r")\b"
    r"|\bhow much\s+(?:is|are|was)\b"
    r"|\bhow much (?:does|do)\b[^?]*\bcost\b",
    _RE_FLAGS,
)

# -- Layer 4: conversational filler -------------------------------------------
_RULE_CHITCHAT_RE = re.compile(
    r"\b("
    r"you (?:said|mentioned|told me)|(?:said|mentioned) (?:earlier|before)|"
    r"repeat that|say that again|come again|what did you say|"
    r"hello|hi there|hey there|good morning|good evening|good afternoon|"
    r"how are you|how's it going|nice to meet you|"
    r"thank you|thanks|goodbye|bye for now|see you|nice day|"
    r"translate|spell|rhymes with|tell me a joke|tell me a story|"
    # Small talk that carries other content, so the greeting-only test above
    # cannot catch it, but that still needs no lookup of any kind.
    r"sing (?:me )?a song|tell me something (?:fun|funny|interesting)|"
    r"let'?s (?:talk|chat)|(?:i'?m|i am) (?:good|fine|great|okay|ok|well|doing)|"
    r"my name is|i'?m called|call me|nice talking|good to hear|"
    r"never ?mind|forget it|no worries|sorry about that|my bad|"
    r"are you sure|really\?|is that (?:right|true|so)|makes sense|"
    r"one (?:moment|second|sec)|hold on|wait a (?:moment|second|sec)"
    r")\b",
    _RE_FLAGS,
)


def rule_route_explain(transcript: str) -> tuple[Optional[bool], str]:
    """Layered rule pass. Returns (verdict, reason).

    verdict is True (search), False (answer directly) or None (undecided ->
    hand it to the model). `reason` names the layer that decided, so a
    surprising route can be traced to one rule instead of guessed at."""
    text = (transcript or "").strip()
    if not text:
        return False, "empty transcript"

    # Layer 0: procedural and advisory shapes outrank even an explicit time
    # word. In "if I invest $1,000 in Bitcoin today, what should I consider?"
    # or "how do I check the gold price today?", the time word attaches to the
    # ACTION, not to a value being requested -- the question is still asking
    # for guidance, and a price lookup does not answer it. Only these two
    # shapes get this precedence: "is Bitcoin up today?" stays a live query
    # because it asks about a movement, not about what to do.
    for label, pattern in _RULE_STATIC_FORMS:
        if label not in ("procedural", "advisory"):
            continue
        m = pattern.search(text)
        if m:
            return False, f"{label} question ({m.group(0).strip()!r}) - asks what to do, not a live value"

    # Layer 0b: the question is about the assistant itself. Outranks the time
    # word below: "how are you today?" and "what's your name today" are small
    # talk, and no web page answers either of them.
    m = _RULE_SELF_RE.search(text)
    if m:
        return False, (
            f"about the assistant itself ({m.group(0).strip()!r}) - answered from its "
            f"own persona, no lookup exists for this"
        )

    # Layer 0c: the utterance is nothing but a greeting or a thank-you. Also
    # above the time word, but only because nothing else was asked -- "good
    # morning, what's the weather today?" keeps falling through to the time
    # layer and still searches.
    greeting = _greeting_only_match(text)
    if greeting:
        return False, f"greeting only ({greeting!r}) - nothing was actually asked"

    m = _RULE_TIME_RE.search(text)
    if m:
        return True, f"explicit time reference {m.group(0)!r}"

    for label, pattern in _RULE_STATIC_FORMS:
        m = pattern.search(text)
        if m:
            return False, f"{label} question ({m.group(0).strip()!r}) - shape needs no lookup"

    m = _RULE_LIVE_TOPIC_RE.search(text)
    if m:
        return True, f"asks for a live value/event ({m.group(0).strip()!r})"

    m = _RULE_CHITCHAT_RE.search(text)
    if m:
        return False, f"conversational phrase ({m.group(0).strip()!r})"

    return None, "no rule matched"


def rule_route(transcript: str) -> Optional[bool]:
    """Instant rule pass. True (search), False (answer directly), or None
    (undecided -> ask the model). Never raises."""
    return rule_route_explain(transcript)[0]


# ── Tier 1: Qwen single-forward-pass router ─────────────────────────────────

_ROUTER_PROMPT = (
    "Decide whether answering this spoken question REQUIRES looking up a fact that "
    "changes over time.\n"
    "\n"
    "Judge the TYPE of question, not its subject. Questions about money, markets, "
    "technology or sport are usually explanatory and need no lookup. Only a request "
    "for a specific current VALUE or EVENT needs one.\n"
    "\n"
    "Answer Yes only when the correct answer would be different tomorrow:\n"
    "- a current price, rate, quote or market value\n"
    "- today's weather\n"
    "- a recent or upcoming event, news story, score or schedule\n"
    "- who currently holds a position\n"
    "- anything dated after your training data\n"
    "\n"
    "Answer No for everything else, including:\n"
    "- what something is, or how it works\n"
    "- how to do something, or what steps to take\n"
    "- whether something is safe, wise, legal or worth doing\n"
    "- benefits, risks, differences, comparisons\n"
    "- history, science, maths, language, code\n"
    "- opinions, recommendations, small talk\n"
    "- ANYTHING about you, the assistant: your name, how you are, who made you, "
    "what you can do, whether you can hear the user\n"
    "\n"
    "Same subject, different answers:\n"
    "Q: Can you tell me your name? -> No (asks about the assistant itself)\n"
    "Q: How are you doing? -> No (small talk)\n"
    "Q: Can you hear me? -> No (asks about the assistant itself)\n"
    "Q: Can you tell me the gold price? -> Yes (asks for a current value)\n"
    "Q: What is the price of Bitcoin? -> Yes (asks for a current value)\n"
    "Q: What is Bitcoin? -> No (asks what something is)\n"
    "Q: How do I invest in Bitcoin? -> No (asks how to do something)\n"
    "Q: Is Bitcoin a safe investment? -> No (asks for a judgement)\n"
    "Q: Did Bitcoin go up today? -> Yes (asks about a recent event)\n"
    "Q: How does the stock market work? -> No (asks for an explanation)\n"
    "Q: How is Tesla stock doing? -> Yes (asks for a current value)\n"
    "\n"
    "Reply with exactly one word: Yes or No.\n"
    "\n"
    "Q: {q}\n"
    "A:"
)


class QueryRouter:
    """Decides 'does this question need a web search?' from the STT transcript.

    Runs ONE forward pass over a small instruct model and compares the logits
    of the 'Yes' and 'No' tokens at the final position. Nothing is sampled and
    nothing is generated, so the call is deterministic, has no stopping-criteria
    edge cases, and costs a single prefill instead of a generation loop.

    The model is shared with `ContextCompressor` (see `from_compressor`) so
    routing adds no extra VRAM and no extra load time -- the compressor sits
    idle except during compression, and the two never run concurrently (routing
    always finishes before a search, and compression only happens after one).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        threshold: float = 0.40,
        use_rules: bool = True,
        max_chars: int = 400,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Deliberately below 0.5. The two errors are not symmetric: searching
        # when it was unnecessary costs ~2s of thinking sound and is
        # recoverable, while NOT searching when it was needed produces a
        # confidently wrong answer spoken aloud. Bias toward searching.
        self.threshold = float(threshold)
        self.use_rules = bool(use_rules)
        self.max_chars = int(max_chars)
        # Prompt length of the most recent _model_score call (0 when the rules
        # resolved the turn and the model never ran). Read by the latency log.
        self.last_prompt_tokens = 0
        self._yes_ids = self._word_token_ids("Yes")
        self._no_ids = self._word_token_ids("No")
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("QueryRouter could not resolve Yes/No token ids for this tokenizer")

    @classmethod
    def from_compressor(
        cls, compressor: "ContextCompressor", threshold: float = 0.40, use_rules: bool = True
    ) -> "QueryRouter":
        return cls(
            model=compressor.model,
            tokenizer=compressor.tokenizer,
            device=compressor.device,
            threshold=threshold,
            use_rules=use_rules,
        )

    def _word_token_ids(self, word: str) -> list[int]:
        """Token ids that can begin `word` at the answer position. Both the
        bare form and the leading-space form are collected: BPE tokenizers emit
        a different id depending on whether the word follows whitespace, and
        checking only one of them silently loses most of the probability mass."""
        ids: list[int] = []
        for variant in (word, " " + word, word.upper(), word.lower(), " " + word.lower()):
            try:
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if enc:
                ids.append(int(enc[0]))
        return sorted(set(ids))

    @torch.inference_mode()
    def _model_score(self, transcript: str) -> float:
        """P(Yes) from a single forward pass."""
        prompt = _ROUTER_PROMPT.format(q=transcript.strip()[: self.max_chars])
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        # Size of the single prefill this router costs -- the only knob that
        # moves its latency, so it belongs next to the measured elapsed time in
        # the latency log.
        self.last_prompt_tokens = int(inputs["input_ids"].shape[1])
        logits = self.model(**inputs).logits[0, -1].float()
        yes_logit = torch.logsumexp(logits[self._yes_ids], dim=0)
        no_logit = torch.logsumexp(logits[self._no_ids], dim=0)
        pair = torch.stack([yes_logit, no_logit])
        return float(torch.softmax(pair, dim=0)[0].item())

    def decide(self, transcript: str) -> dict:
        """Returns {needs_search, source, score, elapsed_s, reason}.

        `source` is 'rules' when Tier 0 resolved it, 'model' when Qwen did, and
        'error' when the model call failed (in which case it fails OPEN -- the
        turn is treated as needing a search, since a wrong answer is worse than
        a slow one)."""
        t0 = time.perf_counter()
        # Cleared up front so a rules-resolved turn never reports the previous
        # turn's prefill size.
        self.last_prompt_tokens = 0
        cleaned = (transcript or "").strip()
        if not cleaned:
            return {
                "needs_search": False, "source": "rules", "score": 0.0,
                "elapsed_s": time.perf_counter() - t0, "reason": "empty transcript",
            }

        if self.use_rules:
            ruled, rule_reason = rule_route_explain(cleaned)
            if ruled is not None:
                return {
                    "needs_search": bool(ruled), "source": "rules",
                    "score": 1.0 if ruled else 0.0,
                    "elapsed_s": time.perf_counter() - t0,
                    "reason": rule_reason,
                }

        try:
            score = self._model_score(cleaned)
        except Exception as e:
            print(f"[search_helpers][router] model scoring failed, defaulting to search: {e!r}", flush=True)
            return {
                "needs_search": True, "source": "error", "score": 1.0,
                "elapsed_s": time.perf_counter() - t0,
                "reason": f"router error ({e!r}); defaulted to searching",
            }

        needs = score >= self.threshold
        return {
            "needs_search": bool(needs), "source": "model", "score": float(score),
            "elapsed_s": time.perf_counter() - t0,
            "reason": (
                f"model P(needs live data)={score:.3f} "
                f"{'>=' if needs else '<'} threshold {self.threshold:.2f}"
            ),
        }


# ── Web search (Tavily / Serper / Bing) ──────────────────────────────────────

# Search providers return raw scraped page text, which is full of things that
# are meaningless when read aloud: markdown headings/tables, image alt-text,
# chart-range buttons ("1D 5D 1M 6M 1Y 5Y"), and marketing boilerplate. Left
# in, this junk reaches the assistant's grounding context verbatim -- confirmed
# in conversation_logs_5, where a gold-price answer was grounded on
# "...back it up with a 120% Best Price Guarantee. Image 13: Dollar
# IconCalculate Gold ValueImage 14: Bell" and the spoken reply became a sales
# pitch for a Seattle gold dealer instead of the price. These patterns strip
# that layer off before the text is scored, compressed, or injected.
_WEB_IMAGE_ALT_RE = re.compile(r"Image\s+\d+\s*:\s*[^\n]*?(?=(?:Image\s+\d+\s*:)|\n|$)")
_WEB_MD_SEPARATOR_RE = re.compile(r"^[\s|:-]*-{2,}[\s|:-]*$")
_WEB_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_WEB_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Matches a whole line of chart-range buttons, whether they arrive one per line
# or space-separated on a single line ("1D 5D 1M 6M 1Y 5Y" -- the shape price
# pages actually scrape as). The previous pattern only allowed ONE token per
# line, so the common single-line run survived cleaning and was read aloud.
# `[ \t]` rather than `\s` on purpose: `\s` matches newlines, which under
# re.MULTILINE would let the pattern span lines and eat real prose.
_WEB_CHART_RANGE_TOKEN = r"(?:\d+[DWMY]|YTD|MAX)"
_WEB_CHART_RANGE_RE = re.compile(
    rf"^[ \t]*{_WEB_CHART_RANGE_TOKEN}(?:[ \t]+{_WEB_CHART_RANGE_TOKEN})*[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
_WEB_BLANKS_RE = re.compile(r"\n{2,}")
_WEB_SPACES_RE = re.compile(r"[ \t]{2,}")


def _flatten_md_table_row(line: str) -> str:
    """Turn '| Gold Price per Ounce | $4,059.41 | £3,041.42 |' into
    'Gold Price per Ounce: $4,059.41, £3,041.42.' -- price/quote pages put the
    answer inside markdown tables, so deleting table rows outright (as an
    earlier version of this cleaner did) threw away the very number the user
    asked for."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cells = [c for c in cells if c]
    if not cells:
        return ""
    if len(cells) == 1:
        return cells[0]
    return f"{cells[0]}: " + ", ".join(cells[1:])


def clean_web_text(text: str, max_chars: int = 1000) -> str:
    """Strip scraped-page furniture from a web search snippet, keeping the
    prose -- and the numbers -- that actually answer questions. Conservative
    by design: it rewrites or removes known-junk structures rather than trying
    to detect 'good' sentences, and returns the original text if cleaning
    would leave almost nothing."""
    if not text:
        return ""
    original = text
    text = _WEB_MD_LINK_RE.sub(r"\1", text)
    text = _WEB_IMAGE_ALT_RE.sub(" ", text)
    text = _WEB_MD_HEADING_RE.sub("", text)
    text = _WEB_CHART_RANGE_RE.sub("", text)
    text = _WEB_BLANKS_RE.sub("\n", text)
    text = _WEB_SPACES_RE.sub(" ", text)
    # Collapse the remaining newlines into sentence-ish separators so the
    # downstream sentence splitter and the compressor prompt both see one
    # continuous piece of prose instead of a column of fragments.
    kept = []
    for ln in (ln.strip() for ln in text.split("\n")):
        if not ln or _WEB_MD_SEPARATOR_RE.match(ln):
            continue
        if ln.startswith("|"):
            ln = _flatten_md_table_row(ln)
            if not ln:
                continue
        # Keep anything carrying letters or digits; drop leftover punctuation-
        # only fragments such as a stray "(" or ")". Digits alone must qualify
        # -- a bare "$298.29" line IS the answer on a stock-quote page.
        if not any(ch.isalnum() for ch in ln):
            continue
        kept.append(ln if ln.endswith((".", "!", "?", ":", ",")) else ln + ".")
    cleaned = " ".join(kept).strip()
    cleaned = _WEB_SPACES_RE.sub(" ", cleaned)
    if len(cleaned) < 40 and len(original.strip()) >= 40:
        # Cleaning removed essentially everything (heavily-structured page);
        # fall back to the raw text rather than discarding a real result.
        return original.strip()[:max_chars]
    return cleaned[:max_chars]


async def web_search_query(
    query: str,
    api_key: Optional[str],
    provider: str,
    max_results: int,
    timeout: float,
) -> list[dict]:
    """Async web search, normalized into a common chunk-dict shape."""
    if not api_key:
        return []

    import aiohttp

    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            if provider == "tavily":
                async with session.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("results", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("url", "web"), "title": r.get("title", ""),
                     "text": clean_web_text(r.get("content", "")),
                     "similarity_score": float(r.get("score", 0.5))}
                    for i, r in enumerate(raw_results)
                ]
            elif provider == "serper":
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("organic", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("link", "web"), "title": r.get("title", ""),
                     "text": clean_web_text(r.get("snippet", "")),
                     "similarity_score": max(0.5, 0.9 - 0.05 * i)}
                    for i, r in enumerate(raw_results)
                ]
            elif provider == "bing":
                async with session.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": api_key},
                    params={"q": query, "count": max_results},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                raw_results = data.get("webPages", {}).get("value", [])[:max_results]
                return [
                    {"id": f"web-{i}", "source": r.get("url", "web"), "title": r.get("name", ""),
                     "text": clean_web_text(r.get("snippet", "")),
                     "similarity_score": max(0.5, 0.9 - 0.05 * i)}
                    for i, r in enumerate(raw_results)
                ]
            else:
                print(f"[search_helpers] unknown web search provider {provider!r}", flush=True)
                return []
    except asyncio.TimeoutError:
        print(f"[search_helpers] web search timed out after {timeout}s: {query!r}", flush=True)
        return []
    except Exception as e:
        print(f"[search_helpers] web search failed: {e!r}", flush=True)
        return []


def web_search_query_sync(query: str, api_key: Optional[str], provider: str, max_results: int, timeout: float) -> list[dict]:
    """Sync wrapper for use from a plain (non-asyncio) background thread."""
    return asyncio.run(web_search_query(query, api_key, provider, max_results, timeout))


# ── Extractive fallback summarizer (no model, no embeddings) ─────────────────

_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for from by with about "
    "as into like through after over between out against during without before "
    "under around among and or but if then than that this these those it its "
    "what which who whom whose when where why how do does did doing done can "
    "could should would may might must will shall have has had i you he she we "
    "they me him her us them my your his our their".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$£€%.,]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


# Abbreviations that must NOT end a sentence. Splitting on them truncated a
# real answer in logs_2 ("The Alphabet Inc. Class C Capital Stock (GOOG) stock
# price today is $339.52" became the headless fragment "Class C Capital Stock
# (GOOG) stock price today is $339.52"), which the assistant then read aloud.
_ABBREV_RE = re.compile(
    r"\b(?:Inc|Ltd|Corp|Co|Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|approx|U\.S|U\.K|E\.U|a\.m|p\.m)\.$",
    re.IGNORECASE,
)
# A figure the question is probably asking for: 4,452.77 / $339.52 / 12% / £3,288
_NUMERIC_FACT_RE = re.compile(r"[$£€¥]\s?\d|\d[\d,]*\.\d|\b\d[\d,]{2,}\b|\d+(?:\.\d+)?\s?%")


def _split_sentences(text: str) -> list[str]:
    """Sentence split that does not break on common abbreviations."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    merged: list[str] = []
    for part in parts:
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [p.strip() for p in merged if p.strip()]


def extract_best_sentence(transcript: str, hits: list[dict], max_chars: int = 200) -> tuple[str, float]:
    """Single best-scoring extractive sentence, plus its confidence score, so
    a caller can decide whether extraction alone is good enough to skip an
    LLM compression pass entirely (that pass cost 2-5s per search turn -- see
    ContextCompressor -- almost all of it GPU contention with the
    concurrently-running PersonaPlex/avatar pipeline).

    Scoring is QUERY COVERAGE (what share of the question's content words the
    sentence addresses) plus a strong bonus for actually containing a figure.
    It is deliberately NOT the `overlap / sqrt(len)` used by
    summarize_web_fallback: that formula rewards shortness so hard that in
    logs_2 it picked the page-title fragment "Gold Price Chart." (score 1.155)
    over the sentence that carried the real answer ("Gold Price per Ounce:
    $4,452.77, ..."), and the assistant then said "Gold Price Chart." out
    loud instead of the price. Fragments shorter than _MIN_ANSWER_CHARS are
    dropped outright for the same reason.

    Returns ("", 0.0) if no usable sentence exists."""
    _MIN_ANSWER_CHARS = 30

    full_text = " ".join(c.get("text", "") for c in hits)
    sentences = _split_sentences(full_text)
    # Prefer properly terminated sentences; scraped passages often end
    # mid-fragment in nav/alt-text junk.
    terminated = [s for s in sentences if s.endswith((".", "!", "?"))]
    if terminated:
        sentences = terminated
    sentences = [s for s in sentences if len(s) >= _MIN_ANSWER_CHARS]
    if not sentences:
        return "", 0.0

    query_words = _content_words(transcript)
    if not query_words:
        return "", 0.0

    scored = []
    for idx, sent in enumerate(sentences):
        sent_words = _content_words(sent)
        if not sent_words:
            continue
        coverage = len(query_words & sent_words) / len(query_words)
        if coverage <= 0:
            continue
        score = coverage
        if _NUMERIC_FACT_RE.search(sent):
            # The overwhelming majority of questions that reach a web search
            # here ask for a value (price/rate/score/"how much"), so a
            # sentence carrying a figure is far more likely to be the answer
            # than a same-topic sentence without one.
            score += 0.40
        # Mild preference for a readable, speakable length: very long scraped
        # runs are worse to read aloud than a focused sentence.
        if len(sent) > 180:
            score -= 0.10
        scored.append((score, -len(sent), idx))
    if not scored:
        return "", 0.0

    scored.sort(reverse=True)
    best_score, _, best_idx = scored[0]
    return sentences[best_idx][:max_chars], float(best_score)


def summarize_web_fallback(
    transcript: str, hits: list[dict], max_sentences: int = 2, max_chars: int = 200,
) -> str:
    """Extractive fallback used only when ContextCompressor is unavailable or
    rejected its own output. Scores sentences by content-word overlap with the
    question -- no embedding model and no generation, so it cannot hallucinate
    and cannot add load time. (The former version of this function scored
    sentences with sentence-transformers; that dependency went away with the
    local-document index, and lexical overlap is adequate for picking between a
    handful of already-relevant web snippets.)"""
    full_text = " ".join(c.get("text", "") for c in hits)
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    # Web-scraped passages frequently end mid-fragment -- the text after the
    # last '.'/'!'/'?' in the source is often nav/alt-text junk with no
    # sentence punctuation of its own (confirmed via conversation_logs_5:
    # a gold-price query pulled in a trailing fragment reading "Image 13:
    # Dollar IconCalculate Gold ValueImage 14: Bell", which scored high on
    # keyword overlap and was read into the assistant's <ref> context
    # verbatim). Prefer properly terminated sentences; only fall back to the
    # unfiltered list if that would leave nothing to summarize from.
    terminated = [s for s in sentences if s.endswith((".", "!", "?"))]
    if terminated:
        sentences = terminated
    if not sentences:
        return full_text[:max_chars]

    query_words = _content_words(transcript)
    scored = []
    for idx, sent in enumerate(sentences):
        sent_words = _content_words(sent)
        if not sent_words:
            continue
        overlap = len(query_words & sent_words)
        # Normalize by sentence length so a long paragraph does not win purely
        # by containing more words than a short, precise answer.
        scored.append((overlap / (len(sent_words) ** 0.5), idx))
    if not scored:
        return " ".join(sentences[:max_sentences])[:max_chars]

    scored.sort(reverse=True)
    chosen = sorted(idx for _, idx in scored[:max_sentences])
    return " ".join(sentences[i] for i in chosen)[:max_chars]


# ── Spoken-English normalization for the text PersonaPlex will actually say ──
#
# The grounding text (extractive, LLM-compressed, or fallback -- all three)
# comes straight from web pages: currency symbols, +/- percent deltas, ticker
# parentheses, markdown, raw URLs. Handed to a SPEECH model verbatim, those
# symbols either get dropped silently or mangled -- and the user's own
# examples ("$309.32" -> should be "three hundred nine dollars and thirty-two
# cents") ask for genuinely spoken-English numbers, not just symbol removal.
# Deterministic, pure Python, no model call: this must never add latency to
# either the search path or (especially) the normal no-search path, which
# never calls it at all.

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand")]
_MAX_SPOKEN_INT = 10 ** 15  # beyond this, spelling out digits stops being natural -- leave as-is


def _int_to_words(n: int) -> str:
    """American short-scale integer -> words. Pure Python, no dependency."""
    if n < 0:
        return f"negative {_int_to_words(-n)}"
    if n < 20:
        return _ONES[n] if n else "zero"
    if n < 100:
        tens, rem = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[rem]}" if rem else "")
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (f" {_int_to_words(rem)}" if rem else "")
    for scale_val, scale_name in _SCALES:
        if n >= scale_val:
            major, rem = divmod(n, scale_val)
            out = f"{_int_to_words(major)} {scale_name}"
            return out + (f" {_int_to_words(rem)}" if rem else "")
    return str(n)  # unreachable given the scale table above


def _spell_digits(digits: str) -> str:
    """Read a digit string one digit at a time -- 'point two three', not
    'point twenty-three'. This is how humans read a decimal fraction aloud."""
    return " ".join(_DIGIT_WORDS[int(ch)] for ch in digits if ch.isdigit())


# sing_major, plur_major, sing_minor, plur_minor
_CURRENCY_WORDS = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "", ""),
}
_COMPACT_SCALE_WORDS = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}

# Compact suffixed amounts FIRST ("$4.20T", "£3,288.91K" if that ever occurs) --
# must run before the plain-amount pattern below, or it would match just the
# "$4.20" prefix and leave a dangling "T". Both money patterns also swallow a
# leading +/- (a price delta, e.g. "-$5.00") so that sign is spoken as
# "up"/"down" rather than surviving as a stray math symbol -- per-instruction,
# a lone hyphen elsewhere in the sentence (a real hyphenated word) is left
# alone since it is never adjacent to a currency symbol or digit.
# No leading `\s?` before the sign/symbol: that would consume whatever
# whitespace precedes the amount (e.g. the space in "is $338.72") as PART of
# the match, and the replacement text (which starts directly with a word, no
# leading space) would then glue onto the previous word -- "is$338.72" ->
# "isthree hundred...". Starting the match exactly at the sign/symbol leaves
# that whitespace untouched in the surrounding text.
_COMPACT_MONEY_RE = re.compile(r"([+-])?([$£€¥])\s?(\d[\d,]*(?:\.\d+)?)\s?([KMBT])\b")
_PLAIN_MONEY_RE = re.compile(r"([+-])?([$£€¥])\s?(\d[\d,]*)(?:\.(\d{1,2}))?")
# +/-1.11%, 0.23%, 12% -- the sign, if present, becomes "up"/"down" rather
# than being dropped, since it is often the actual point of the sentence
# ("reflecting a +0.23% move").
_PERCENT_RE = re.compile(r"([+-])?\s?(\d+)(?:\.(\d+))?\s?%")
# Ticker parentheticals: (TSLA), (GOOG), (GOOGL), (BRK.A) -- short, all-caps,
# optionally dotted. Deliberately narrow so it never eats a genuine
# parenthetical remark, which is never all-caps/short like this.
_TICKER_PAREN_RE = re.compile(r"\s?\([A-Z]{1,6}(?:\.[A-Z]{1,3})?\)")
_URL_RE = re.compile(r"https?://\S+")
_MARKDOWN_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_EMPTY_PAREN_RE = re.compile(r"\(\s*\)")
# Anything with a currency symbol or a "%" is handled (and its sign consumed)
# above. This is the leftover case: a bare signed number with neither, e.g.
# "337.12 +2.10" from a raw index quote. Not preceded by a word char or a
# decimal point, so it never touches a real hyphenated word or a negative
# exponent -- only a sign standing immediately in front of a digit.
_STRAY_SIGNED_NUMBER_RE = re.compile(r"(?<![\w.])([+-])\s?(?=\d)")
# A lone +/- used as a bullet/separator with no adjacent number at all.
_STRAY_MATH_SYMBOL_RE = re.compile(r"(?<=\s)[+-](?=\s)")
# "US$146.19", "A$50", "C$12.30" -- an ISO-ish letter prefix glued to the
# symbol. Left in place it strands against the replacement words, which is
# how logs_4 turn 3 produced "USone hundred forty-six dollars and nineteen
# cents". Keep the symbol (the money patterns still need it), drop the letters.
_CURRENCY_LETTER_PREFIX_RE = re.compile(r"\b(?:US|AU|NZ|CA|C|A|HK|SG)(?=[$£€¥]\s?\d)")
# An article stranded in front of an up/down that replaced a +/- sign:
# "reflecting a +0.23% move" -> "reflecting a up 0.23 percent move".
_STRANDED_ARTICLE_RE = re.compile(r"\b[Aa]n?\s+(?=(?:up|down)\b)")
# ...and the noun that only made sense attached to the symbol form.
_DANGLING_MOVE_RE = re.compile(r"\bpercent\s+(?:move|change|gain|loss)\b")


def _compact_money_sub(m: "re.Match[str]", spell_numbers: bool) -> str:
    sign, symbol, num, suffix = m.group(1), m.group(2), m.group(3).replace(",", ""), m.group(4)
    sign_word = {"+": "up ", "-": "down "}.get(sign or "", "")
    scale_word = _COMPACT_SCALE_WORDS[suffix]
    _, plur_major, _, _ = _CURRENCY_WORDS.get(symbol, _CURRENCY_WORDS["$"])
    if not spell_numbers:
        return f"{sign_word}{num} {scale_word} {plur_major}"
    if "." in num:
        whole_s, frac_s = num.split(".", 1)
        num_words = f"{_int_to_words(int(whole_s or 0))} point {_spell_digits(frac_s)}"
    else:
        num_words = _int_to_words(int(num))
    return f"{sign_word}{num_words} {scale_word} {plur_major}"


def _plain_money_sub(m: "re.Match[str]", spell_numbers: bool) -> str:
    sign, symbol, whole_s, frac_s = m.group(1), m.group(2), m.group(3).replace(",", ""), m.group(4)
    sign_word = {"+": "up ", "-": "down "}.get(sign or "", "")
    sing_major, plur_major, sing_minor, plur_minor = _CURRENCY_WORDS.get(symbol, _CURRENCY_WORDS["$"])
    whole = int(whole_s)
    if whole >= _MAX_SPOKEN_INT:
        return m.group(0)  # absurdly large -- leave the digits alone
    major_word = sing_major if whole == 1 else plur_major
    if not spell_numbers:
        # Keep the numeral EXACTLY as written, drop only the symbol. See
        # normalize_for_speech's docstring for the logs_3-vs-logs_4 evidence
        # that spelling the digits out is what corrupted $309.32 into $39.32.
        amount = f"{whole_s}.{frac_s}" if frac_s else whole_s
        return f"{sign_word}{amount} {major_word}"
    phrase = f"{sign_word}{_int_to_words(whole)} {major_word}"
    if frac_s and sing_minor:
        cents = int(frac_s.ljust(2, "0")[:2])
        if cents:
            minor_word = sing_minor if cents == 1 else plur_minor
            phrase += f" and {_int_to_words(cents)} {minor_word}"
    return phrase


def _percent_sub(m: "re.Match[str]", spell_numbers: bool) -> str:
    sign, whole_s, frac_s = m.group(1), m.group(2), m.group(3)
    sign_word = {"+": "up ", "-": "down "}.get(sign or "", "")
    if not spell_numbers:
        amount = f"{whole_s}.{frac_s}" if frac_s else whole_s
        return f"{sign_word}{amount} percent"
    whole_words = _int_to_words(int(whole_s))
    if frac_s:
        return f"{sign_word}{whole_words} point {_spell_digits(frac_s)} percent"
    return f"{sign_word}{whole_words} percent"


def normalize_for_speech(text: str, spell_numbers: bool = False) -> str:
    """Rewrite web-search grounding text into plain spoken English before it
    is handed to PersonaPlex, so the model speaks natural language instead of
    reading symbols aloud (or silently dropping them).

    Deterministic and pure-Python -- no model call, ~microseconds.

    `spell_numbers` defaults to **False**, against the obvious intuition,
    because two real RunPod runs A/B'd it by accident and the digits won:

      logs_3 turn 6, ref said "...price today is $309.32, reflecting a
      +0.23% move..."  -> the assistant SPOKE "$309.32, up 0.23%".  Correct.
      logs_4 turn 4, ref said "...is three hundred nine dollars and
      thirty-two cents..." -> the assistant SPOKE "$39.32".  Wrong number.

    Spelling the digits out roughly doubles the token count of the injected
    <ref> (36 -> 42 tokens for the same fact) and turns one unambiguous
    numeral into a sequence the model has to RE-ASSEMBLE. When the injection
    smears (see the ref-drain notes in imtalker_personaplex_try_vad2_8998.py)
    it reassembled "three ... nine ... thirty-two" while dropping the
    "hundred" magnitude entirely. A shorter ref also smears less and injects
    faster, so keeping the numeral is better on accuracy AND latency.

    What still always happens (these are pure wins, no downside): the
    currency/percent SYMBOLS themselves become words ($ -> dollars,
    % -> percent, +/- -> up/down), ticker parentheses, raw URLs, markdown and
    table pipes are stripped, and the article/filler grammar left behind by
    those substitutions is repaired. PersonaPlex re-verbalizes the numeral in
    its own voice anyway -- logs_4 shows it saying "up 0.23%" no matter which
    form it was given -- so the numeral's written form costs nothing spoken.

    Pass spell_numbers=True to get the fully spelled-out form
    ("three hundred nine dollars and thirty-two cents").
    """
    if not text:
        return text
    out = text
    out = _URL_RE.sub("", out)
    out = _MARKDOWN_TABLE_PIPE_RE.sub(", ", out)
    out = _SYMBOL_RE.sub("", out)  # markdown emphasis/heading/code marks
    # "US$146.19" / "A$50" / "C$12.30": strip the ISO-ish letter prefix that
    # would otherwise be left stranded against the substituted words
    # ("USone hundred forty-six dollars..." -- seen in logs_4 turn 3).
    out = _CURRENCY_LETTER_PREFIX_RE.sub("", out)
    out = _COMPACT_MONEY_RE.sub(lambda m: _compact_money_sub(m, spell_numbers), out)
    out = _PLAIN_MONEY_RE.sub(lambda m: _plain_money_sub(m, spell_numbers), out)
    out = _PERCENT_RE.sub(lambda m: _percent_sub(m, spell_numbers), out)
    out = _STRAY_SIGNED_NUMBER_RE.sub(lambda m: {"+": "up ", "-": "down "}[m.group(1)], out)
    out = _STRAY_MATH_SYMBOL_RE.sub(",", out)
    out = _TICKER_PAREN_RE.sub("", out)
    out = _EMPTY_PAREN_RE.sub("", out)
    # -- Grammar repair for what the substitutions above leave behind --------
    # "reflecting a +0.23% move" becomes "reflecting a up 0.23 percent move":
    # the article and the trailing noun were grammatical only around the
    # symbol. Drop the stranded article, then the now-dangling "move".
    out = _STRANDED_ARTICLE_RE.sub("", out)
    out = _DANGLING_MOVE_RE.sub("percent", out)
    # Collapse whitespace/punctuation artifacts left behind by the removals
    # above (double spaces, a space before a period, doubled commas).
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,!?])", r"\1", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r",\s*\.", ".", out)
    return out.strip()


# ── Context compressor: small LLM, query + top-k hits -> 1-2 sentence grounding ──

class ContextCompressor:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 40,
        quantize_4bit: bool = True,
        max_passages: int = 2,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

        print(f"[search_helpers][compressor] loading {model_name} on {device} (4bit={quantize_4bit}) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        common_kwargs = dict(token=os.getenv("HF_TOKEN"), attn_implementation="sdpa")
        if quantize_4bit and device != "cpu":
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, device_map=device, **common_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device, **common_kwargs,
            )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_passages = max_passages
        self.eos_ids = [self.tokenizer.eos_token_id]
        # Filled in by every compress() call and read by the caller for the
        # latency log ("how many tokens did the compressor output?"). Kept as
        # an attribute rather than changing compress()'s return type, which is
        # consumed as a plain string in several places.
        self.last_stats: dict = {}

        class _SentenceEndCriteria(StoppingCriteria):
            def __init__(self, tokenizer, prompt_len, min_new=6):
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.min_new = min_new

            def __call__(self, input_ids, scores, **kwargs):
                new_len = input_ids.shape[1] - self.prompt_len
                if new_len < self.min_new:
                    return False
                tail = self.tokenizer.decode(input_ids[0, -3:]).rstrip()
                if not tail.endswith((".", "!", "?", "\n")):
                    return False
                # A '.' between digits is a decimal point, not a sentence end.
                # Without this check, generation stopped inside prices --
                # confirmed in conversation_logs_4, where the gold price
                # $4,068.60 was cut to "$4,068." before the cents were
                # generated.
                if re.search(r"\d[.,]\d*$", tail):
                    return False
                return True

        self._StoppingCriteriaList = StoppingCriteriaList
        self._SentenceEndCriteria = _SentenceEndCriteria
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[search_helpers][compressor] ready — {n_params:.2f}B params", flush=True)

    def compress(self, question: str, chunks: list[dict]) -> str:
        self.last_stats = {
            "compressor_prompt_tokens": 0,
            "compressor_generated_tokens": 0,
            "compressor_output_tokens": 0,
            "compressor_output_chars": 0,
            "compressor_max_new_tokens": int(self.max_new_tokens),
            "compressor_hit_token_cap": False,
            "compressor_generate_s": 0.0,
            "compressor_rejected": "",
        }
        if not chunks:
            return ""
        # 320 chars, not 180: the old limit routinely cut the passage off
        # before the sentence carrying the actual figure. `max_passages` is
        # now honored as configured -- it was previously clamped by
        # min(max_passages, 2), so raising the setting silently did nothing.
        passages = "\n".join(
            f"[{i + 1}] {c['text'][:320]}" for i, c in enumerate(chunks[: self.max_passages])
        )
        user_content = (
            "You are given passages from a web page or document. Answer the question in "
            "ONE short sentence that will be read aloud.\n"
            "Rules:\n"
            "- State the specific fact asked for (the number, price, date, or name) and "
            "include its units or currency.\n"
            "- Ignore advertising, slogans, menus, image captions, and any text about the "
            "website or seller itself -- it is page furniture, not an answer.\n"
            "- Plain text only: no markdown, no lead-in phrase, no citation markers, no ticker "
            "symbols in parentheses like (TSLA).\n"
            "- Write it the way a person would SAY it out loud, not how it is printed: numbers "
            "and prices are still fine as digits ($309.32, 12%), just avoid raw table/list "
            "formatting and symbols that only make sense written down.\n"
            "- Never reply conversationally. You are writing a fact for someone else to say, "
            "not talking to the user: never answer with \"Yes, I can...\", an offer to help, or "
            "a comment about yourself. If the question is phrased as a yes/no request such as "
            "\"can you tell me about X\", state the key fact about X instead.\n"
            "- If the passages genuinely do not answer the question, reply exactly NO_CONTEXT.\n"
            f"Q: {question}\nPassages:\n{passages}\nA:"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        stopping = self._StoppingCriteriaList([self._SentenceEndCriteria(self.tokenizer, prompt_len)])
        t_gen0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_ids,
                stopping_criteria=stopping,
            )
        gen_elapsed = time.perf_counter() - t_gen0
        new_tokens = out[0, prompt_len:]
        n_new = int(new_tokens.shape[0])
        # Recorded before any of the cleanup below can reject the answer, so a
        # rejected generation still shows what it cost. `hit_token_cap` is the
        # one that matters when an answer is cut off mid-figure: it means the
        # sentence-end stopping criterion never fired and max_new_tokens did.
        self.last_stats.update({
            "compressor_prompt_tokens": int(prompt_len),
            "compressor_generated_tokens": n_new,
            "compressor_hit_token_cap": bool(n_new >= int(self.max_new_tokens)),
            "compressor_generate_s": round(gen_elapsed, 4),
        })
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result = _SYMBOL_RE.sub("", result).strip()
        # Strip a leading list marker ("1. ", "- ", "• ") only when it is
        # followed by whitespace. The previous pattern (^[-•\d.)]+\s*) ate any
        # leading digits, so an answer that opened with a bare number -- the
        # normal shape of a price or stock-quote answer -- was silently
        # mangled before it ever reached the caller.
        result = re.sub(r"^\s*(?:[-•*]|\d{1,2}[.)])\s+", "", result)
        # Token count of what the compressor actually produced, measured on the
        # cleaned text that the caller will use -- not on the raw generation,
        # which still carries the stop token and any list marker stripped above.
        try:
            n_out_tokens = len(self.tokenizer.encode(result, add_special_tokens=False)) if result else 0
        except Exception:
            n_out_tokens = 0
        self.last_stats.update({
            "compressor_output_tokens": n_out_tokens,
            "compressor_output_chars": len(result),
        })

        if "NO_CONTEXT" in result or not result:
            print(f"[search_helpers][compressor] rejected (NO_CONTEXT/empty): {result!r}", flush=True)
            self.last_stats["compressor_rejected"] = "no_context_or_empty"
            return ""

        passage_words = set(w.lower() for c in chunks[: self.max_passages] for w in c["text"].split())
        answer_words = set(w.lower().strip(".,!?") for w in result.split())
        overlap = len(answer_words & passage_words) / max(1, len(answer_words))
        self.last_stats["compressor_passage_overlap"] = round(overlap, 3)
        if overlap < 0.15:
            print(
                f"[search_helpers][compressor] rejected (low overlap={overlap:.2f}): {result!r}",
                flush=True,
            )
            self.last_stats["compressor_rejected"] = f"low_overlap={overlap:.2f}"
            return ""
        return result


# ── Isolated upstream-`moshi` loader (for the STT/VAD submodel only) ────────
#
# Both the PersonaPlex fork this project already runs (`liveTry.py`'s
# `import moshi`, resolved via `_ensure_moshi_importable`) and the real
# upstream Kyutai `moshi` PyPI package (needed here for
# `moshi.models.loaders.CheckpointInfo`, which the fork's own loaders.py does
# not define) both package themselves under the same top-level import name
# `moshi`. They cannot both occupy `sys.modules["moshi"]`. This loads the
# upstream package from an isolated install directory under a private alias
# (`moshi_stt`) via importlib, so it never touches `sys.modules["moshi"]` and
# never fights with the PersonaPlex fork for the name.

def load_upstream_moshi_stt(stt_pkg_dir: str):
    """Load the genuine upstream Kyutai `moshi` PyPI package (installed via
    `pip install --no-deps --target <stt_pkg_dir> moshi`) under the private
    module name `moshi_stt`. Returns the loaded module, or raises on failure
    -- callers should wrap this in try/except and disable STT/search on
    failure, never let it block avatar startup.
    """
    if "moshi_stt" in sys.modules:
        return sys.modules["moshi_stt"]

    init_path = Path(stt_pkg_dir) / "moshi" / "__init__.py"
    if not init_path.is_file():
        raise FileNotFoundError(
            f"upstream moshi package not found at {init_path} -- run "
            f"`pip install --no-deps --target {stt_pkg_dir} moshi` first"
        )
    spec = importlib.util.spec_from_file_location(
        "moshi_stt", init_path, submodule_search_locations=[str(init_path.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["moshi_stt"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("moshi_stt", None)
        raise
    return module
