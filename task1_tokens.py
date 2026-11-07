"""
Lab 6 - Task 1 (Beginner): Tokenization and a FAQ Chatbot
Run: python task1_tokens.py

Part A: compare whitespace split vs Unicode word-like regex (+ model tokenizer if available)
Part B: TF-IDF FAQ chatbot with two added FAQs, each with a source
All FAQ text below is synthetic teaching data.
"""
import re

# =====================================================================
# PART A - Tokenization comparison
# =====================================================================
samples = [
    "AI, search!",              # punctuation
    "I can't log in.",          # contraction
    "help@example.com",         # email
    "CS4944 version 2.1",       # identifier + number
    "",                         # empty string
    "یہ ایک مثال ہے",             # Urdu text
]

print("=" * 70)
print("PART A: TOKENIZATION")
print("=" * 70)
for text in samples:
    ws = text.split()
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print("Original   :", repr(text))
    print("  Whitespace:", ws, "->", len(ws), "tokens")
    print("  Word-like :", words, "->", len(words), "tokens")
assert re.findall(r"\b\w+\b", "") == []   # empty input produces no units

# Optional model sub-word tokenizer (needs the package AND a first-time download)
print("\n--- Model tokenizer (all-MiniLM-L6-v2), two English examples ---")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for text in samples[:2]:
        print(repr(text), "->", model.tokenizer.tokenize(text))
except Exception as e:  # ImportError, no internet, etc.
    print("Model tokenizer NOT available in this run:", type(e).__name__)
    print("(Install sentence-transformers and allow the first model download to see it.)")

# =====================================================================
# PART B - FAQ chatbot (TF-IDF retrieval + confirmation workflow)
# =====================================================================
from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever:
    """Minimal TF-IDF retriever (same logic as the lab's lab6_search.py)."""

    def __init__(self, records):
        if not records or any(not r["text"].strip() for r in records):
            raise ValueError("Provide nonempty records")
        self.records = records
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    def search(self, query, k=3):
        if not query.strip() or k <= 0:
            return []
        q = self.vectorizer.transform([query])
        if q.nnz == 0:                      # no vocabulary overlap -> no matches
            return []
        scores = (q @ self.matrix.T).toarray()[0]
        ranked = sorted(range(len(self.records)), key=lambda i: (-float(scores[i]), i))
        return [(self.records[i], float(scores[i])) for i in ranked[:k]]


FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.", "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    # ---- two FAQs added for this task (synthetic) ----
    {"id": "faq:wifi", "question": "How do I connect to the campus Wi-Fi?",
     "answer": "Select the campus network and sign in with your student ID.",
     "source": "wifi_help.txt"},
    {"id": "faq:library", "question": "What are the library opening hours?",
     "answer": "The library is open from 8 am to 8 pm on weekdays.",
     "source": "library_info.txt"},
]


def build_bot():
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records)


def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    # Only the exact canonical FAQ question is answered automatically.
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append("  - " + r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


print("\n" + "=" * 70)
print("PART B: FAQ CHATBOT TRANSCRIPT (TF-IDF)")
print("=" * 70)
engine = build_bot()
demo = [
    ("Exact answer (original FAQ)",      "How do I reset my password?"),
    ("Exact answer (added FAQ 1)",       "How do I connect to the campus Wi-Fi?"),
    ("Exact answer (added FAQ 2)",       "What are the library opening hours?"),
    ("Suggestion (non-exact question)",  "password recovery"),
    ("Suggestion (non-exact, added FAQ)", "library hours"),
    ("Empty input",                      ""),
    ("Unsupported input",                "volcano geology"),
]
for label, q in demo:
    print(f"\n[{label}]")
    print("You:", repr(q))
    print("Bot:", reply(engine, q))

# Checks (same ones as the lab's task1_chatbot_check.py, plus the new FAQs)
assert "it_help.txt" in reply(engine, "How do I reset my password?")
assert "wifi_help.txt" in reply(engine, "How do I connect to the campus Wi-Fi?")
assert "library_info.txt" in reply(engine, "What are the library opening hours?")
assert reply(engine, "") == "Please enter a question."
assert "No matching FAQ" in reply(engine, "volcano geology")
assert "Possible matches" in reply(engine, "password recovery")
print("\nFAQ behaviour checks passed")
