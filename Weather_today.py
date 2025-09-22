#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WSJ Markets Brief (10AM ET) — feed ALL items at once
- Fetch ALL recent WSJ Markets / US Business RSS items within HOURS_BACK (no caps, no keyword filters)
- Provide title + combined summary/description for EVERY item to ChatGPT in one prompt
- Model writes 2–5 numbered paragraphs (1), (2), …; each 2–3 sentences, no repetition across paragraphs
- Inline citations must refer to the evidence list [1]..[N]; code normalizes, validates, and re-numbers cited items to clean 1..K
- References mirror the final sequential numbering exactly (1..K), in order of first citation
- Email marks "GPT summarize:" or fallback reason
"""

import os, ssl, smtplib, feedparser, html, re, logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from dateutil import parser as dtparse
import pytz

# ---------------- Settings ----------------
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
WSJ_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
    "https://feeds.content.dowjones.io/public/rss/RSSWSJD",
    "https://feeds.content.dowjones.io/public/rss/socialeconomyfeed",
    "https://seekingalpha.com/api/sa/combined/TSLA.xml",
]
HOURS_BACK = int(os.getenv("HOURS_BACK", "26"))
INCLUDE_WEEKENDS = os.getenv("INCLUDE_WEEKENDS", "false").lower() == "true"

# Delivery
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD")
TO_EMAIL   = os.getenv("TO_EMAIL", GMAIL_USER)
FROM_NAME  = os.getenv("FROM_NAME", "WSJ Markets Brief")

# LLM
USE_LLM = True
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s %(message)s")

USED_LLM = False
FALLBACK_REASON = None

# ---------------- Helpers ----------------
tz = pytz.timezone(TIMEZONE)

def now_et():
    return datetime.now(tz)

def within_window(dt):
    cutoff = now_et() - timedelta(hours=HOURS_BACK)
    return dt >= cutoff

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", "", t)  # strip HTML tags if any
    return t.strip()

def parse_pubdate(entry):
    for fld in ("published", "updated", "created"):
        val = getattr(entry, fld, None)
        if val:
            try:
                dt = dtparse.parse(val)
                if not dt.tzinfo:
                    dt = pytz.utc.localize(dt)
                return dt.astimezone(tz)
            except Exception:
                pass
    return now_et()

def fetch_items():
    """Fetch ALL recent items within HOURS_BACK; de-dup by link; newest first."""
    items = []
    for url in WSJ_FEEDS:
        logging.info(f"Fetching: {url}")
        feed = feedparser.parse(url)
        if getattr(feed, "bozo", 0):
            logging.warning(f"Feed parse warning for {url}: {getattr(feed, 'bozo_exception', '')}")
        for e in feed.entries:
            title = clean_text(getattr(e, "title", "")) or ""
            link  = getattr(e, "link", "") or ""
            summary = clean_text(getattr(e, "summary", "")) or ""
            description = clean_text(getattr(e, "description", "")) or ""
            # Combine summary + description (avoid duplicate text)
            combined = summary
            if description and description not in summary:
                combined = (summary + " " + description).strip() if summary else description
            pub = parse_pubdate(e)
            if not title or not link:
                continue
            if not INCLUDE_WEEKENDS and pub.weekday() >= 5:
                continue
            if within_window(pub):
                items.append({"title": title, "link": link, "abstract": combined, "pub": pub})
    # de-dup by link; newest first
    uniq = {it["link"]: it for it in items}
    return sorted(uniq.values(), key=lambda x: x["pub"], reverse=True)

# --------- Citation & text utilities ----------
def normalize_brackets(text: str) -> str:
    """Convert {n} or (n) used as citations into [n]."""
    text = re.sub(r"\{(\d+)\}", r"[\1]", text)
    text = re.sub(r"\((\d+)\)", r"[\1]", text)
    return text

def strip_invalid_citations(text: str, max_n: int) -> str:
    """Remove any [n] with n outside 1..max_n to avoid dangling refs."""
    def repl(m):
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= max_n else ""
    return re.sub(r"\[(\d+)\]", repl, text)

def extract_ref_order(summary_text: str, max_n: int):
    """Unique [n]s in order of first appearance, constrained to 1..max_n."""
    nums = []
    for m in re.finditer(r"\[(\d+)\]", summary_text):
        n = int(m.group(1))
        if 1 <= n <= max_n and n not in nums:
            nums.append(n)
    return nums

def renumber_citations(summary_text: str, cited_nums):
    """Map old numbers (in first-appearance order) to new sequential 1..K; return (new_text, mapping old->new)."""
    mapping = {old: i+1 for i, old in enumerate(cited_nums)}
    def repl(m):
        old = int(m.group(1))
        return f"[{mapping[old]}]" if old in mapping else ""
    new_text = re.sub(r"\[(\d+)\]", repl, summary_text)
    return new_text, mapping

def html_paragraphs(text: str) -> str:
    """Turn blank-line-separated blocks into <p> tags."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return "".join(f"<p style='margin:8px 0; font-family:Arial,Helvetica,sans-serif; line-height:1.6;'>{html.escape(p)}</p>" for p in parts)

# ---------------- LLM summarization ----------------
def summarize_overall(numbered_items):
    """
    Feed ALL items at once.
    Require 2–5 numbered paragraphs ('1)', '2)', …), each a distinct firm/theme (no repetition).
    Allow unlimited chained citations referring ONLY to evidence [1]..[N].
    """
    global USED_LLM, FALLBACK_REASON

    n = len(numbered_items)
    if n == 0:
        USED_LLM = False
        FALLBACK_REASON = "NO_ITEMS"
        return "No fresh market-moving headlines in the last 24 hours."

    # Build evidence list from ALL items
    evidence = "\n".join(
        f"[{i+1}] {it['title']} ({it['link']}): {it['abstract']}"
        for i, it in enumerate(numbered_items)
    )

    if not USE_LLM or not OPENAI_API_KEY:
        USED_LLM = False
        FALLBACK_REASON = "NO_API_KEY" if not OPENAI_API_KEY else "USE_LLM_FALSE"
        # simple fallback: first two titles
        return (
            "1) " + (numbered_items[0]['title'] if n >= 1 else "No item") + "\n\n"
            "2) " + (numbered_items[1]['title'] if n >= 2 else "")
        ).strip()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Role: You are a professional U.S. financial-markets analyst.\n\n"
            "Goal: From the article list below, include ONLY items that are materially relevant to "
            "U.S. financial markets OR AI-related firms (like Tesla, Nvidia, OpenAI) OR corporate culture OR financial analysts"
            "Ignore everything else.\n\n"
            "Output rules:\n"
            "1) Write 3–5 short, numbered paragraphs as '1)', '2)', '3)'. "
            "   Each paragraph must cover one DISTINCT firm or macro theme—do not repeat a firm/theme across paragraphs.\n"
            "2) Each paragraph should be 2–3 sentences; each paragraph has about 30 words; neutral tone; factual; no quotes/judgement/conclusion.\n"
            f"3) Use inline numeric citations [1]..[{n}] that refer ONLY to the evidence list below (do not invent numbers). "
            "   Place each citation at the end of the clause it supports. If multiple articles support a point, "
            "   chain ALL relevant citations like [3][7][12]. There is NO upper limit on citations per paragraph, but make sure no repeatitive citations.\n"
            "4) Do NOT re-number sources; keep the evidence numbers exactly as provided. "
            "   (The system will remap cited sources to 1..K later.)\n\n"
            f"{evidence}"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Be concise, objective, and market-focused."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        USED_LLM = True
        FALLBACK_REASON = None
        return resp.choices[0].message.content.strip()
    except Exception as e:
        USED_LLM = False
        FALLBACK_REASON = f"OPENAI_ERROR: {e}"
        logging.warning(f"LLM failed; fallback. Reason: {FALLBACK_REASON}")
        return (
            "1) " + (numbered_items[0]['title'] if n >= 1 else "No item") + "\n\n"
            "2) " + (numbered_items[1]['title'] if n >= 2 else "")
        ).strip()

# ---------------- Email build/send ----------------
def build_email(items):
    date_str = now_et().strftime("%A, %b %d, %Y")

    # Use ALL fetched items; number once (1..N) in newest-first order
    numbered = list(items)

    # Summarize and clean citations
    overall_raw = summarize_overall(numbered)
    overall = normalize_brackets(overall_raw)
    overall = strip_invalid_citations(overall, len(numbered))

    # Extract cited evidence numbers (old) in first-appearance order
    old_order = extract_ref_order(overall, len(numbered))

    # Re-number to clean 1..K and build references accordingly
    if old_order:
        overall, mapping = renumber_citations(overall, old_order)
        refs_html_parts, refs_text_parts = [], []
        for old, new in sorted(mapping.items(), key=lambda kv: kv[1]):
            it = numbered[old - 1]
            refs_html_parts.append(f"[{new}] <a href='{it['link']}'>{html.escape(it['title'])}</a>")
            refs_text_parts.append(f"[{new}] {it['title']} {it['link']}")
    else:
        mapping = {}
        refs_html_parts = ["No references available."]
        refs_text_parts = ["No references available."]

    refs_html = "<br>".join(refs_html_parts)
    refs_text = "\n".join(refs_text_parts)

    origin_label = "GPT summarize:" if USED_LLM else f"Fallback summary (reason: {FALLBACK_REASON}):"

    # Render numbered paragraphs nicely
    def html_paragraphs(text: str) -> str:
        parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        # Bold the leading "1) ", "2) " for readability
        def stylize(s: str) -> str:
            return re.sub(r"^(\d\))\s*", r"<strong>\1</strong> ", html.escape(s))
        return "".join(f"<p style='margin:8px 0; font-family:Arial,Helvetica,sans-serif; line-height:1.6;'>{stylize(p)}</p>" for p in parts)

    html_summary = html_paragraphs(overall)

    # Plain text
    text_body = (
        f"WSJ Markets Brief — {date_str}\n\n"
        f"{origin_label}\n{overall}\n\n"
        f"References:\n{refs_text}"
    )

    # HTML
    html_body = f"""
    <html><body>
      <h2 style="margin:0 0 10px;">WSJ Markets Brief — {date_str}</h2>
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6;">
        <strong>{origin_label}</strong>
      </div>
      {html_summary}
      <h4 style="margin-top:14px;">References</h4>
      <div style="font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.6;">
        {refs_html}
      </div>
      <hr style="margin-top:16px;">
      <div style="font-size:12px;color:#888;">
        Source: WSJ RSS feeds (headlines/abstracts only). This email summarizes permitted feed fields.
      </div>
    </body></html>
    """
    return text_body, html_body

def send_email(text_body, html_body):
    if not GMAIL_USER or not GMAIL_PASS:
        raise RuntimeError("GMAIL_USER and GMAIL_APP_PASSWORD must be set.")
    pw = GMAIL_PASS.replace(" ", "").replace("\u00a0", "").strip()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"WSJ Markets Brief — {now_et().strftime('%b %d')}"
    msg["From"] = f"{FROM_NAME} <{GMAIL_USER}>"
    msg["To"] = TO_EMAIL

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls(context=ctx)
        server.login(GMAIL_USER, pw)
        server.sendmail(GMAIL_USER, [TO_EMAIL], msg.as_string())

# ---------------- Main ----------------
if __name__ == "__main__":
    logging.info("Starting WSJ Markets Brief job…")
    items = fetch_items()  # ALL items; we do not trim
    logging.info(f"Fetched {len(items)} items (pre-filter).")
    text_body, html_body = build_email(items)
    send_email(text_body, html_body)
    logging.info(f"Sent brief to {TO_EMAIL}. USED_LLM={USED_LLM} FALLBACK_REASON={FALLBACK_REASON}")
