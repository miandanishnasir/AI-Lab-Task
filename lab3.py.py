# lab6_all.py
# Lab 6: Tokenization, Embeddings and Text Chunking
# Name: Rasikh Ali - 42365
#
# This ONE file has all the code of the lab.
# All text in this file is made-up teaching data (not real policies).
#
# How to run (pick one):
#   python lab6_all.py tokens       -> Task 1: compare token splitting
#   python lab6_all.py faq          -> Task 1: FAQ chatbot (type questions)
#   python lab6_all.py scenarios    -> the 5 scenarios
#   python lab6_all.py metrics      -> Hit@3 and MRR@3 test
#   python lab6_all.py chunks       -> Task 2: chunking tests and comparison
#   python lab6_all.py retrieval    -> Task 3: full evaluation (makes CSV)
#   python lab6_all.py doc          -> Task 3: document chatbot (type questions)
#   python lab6_all.py all          -> runs tokens, scenarios, metrics, chunks, retrieval
#
# Needs: pip install numpy scikit-learn
# Optional (for embeddings): pip install sentence-transformers

import csv
import json
import sys
import time
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ======================================================================
# PART 1: DATA (5 made-up documents and 10 questions)
# ======================================================================

DOCS = {
    "late": (
        "Assessment Policy for Undergraduate Courses. All coursework must be "
        "submitted before the posted deadline, and late submission is normally "
        "penalized by ten percent for each day after the deadline. Students with "
        "illness may request an extension with medical evidence before the "
        "deadline. The request must be sent to the course coordinator, who will "
        "record the decision and inform the student by email within two working "
        "days of receiving the request. Lab reports follow a separate rule: "
        "students with illness who miss a lab report deadline must provide a "
        "doctor's note within three days after the deadline, and the coordinator "
        "may then approve a short extension only if the note covers the date of "
        "the missed deadline. Group projects are marked as a whole, so one "
        "member's extension does not automatically change the deadline for the "
        "other members unless the coordinator approves a revised schedule in "
        "writing for the whole group and records the change in the course "
        "portal. Students should keep copies of all emails and receipts related "
        "to any request. Unapproved late work after seven days is not marked."),
    "password": (
        "IT Help Desk Guide: Accounts and Passwords. Every student receives a "
        "university account that gives access to email, the learning portal and "
        "campus WiFi. A forgotten password can be reset using the recovery link "
        "on the login page. The link sends a one-time code to the registered "
        "mobile number, and the code expires after ten minutes, so it should be "
        "used straight away. Never share recovery codes with anyone, including "
        "staff, because the help desk will never ask for them. If the registered "
        "mobile number is no longer active, students must visit the IT counter "
        "with their student card so that staff can update the record before any "
        "reset is possible. Staff accounts follow a different process: staff "
        "passwords are reset only by the system administrator after identity "
        "verification, and the recovery link is not available for staff "
        "accounts. Passwords must contain at least twelve characters, and they "
        "should be changed every six months. Accounts are locked for thirty "
        "minutes after five failed login attempts, and the lock cannot be "
        "removed early by the help desk."),
    "assignment": (
        "Assignment Submission Instructions. This guide explains how students "
        "deliver coursework in the current semester. Upload assignments through "
        "the learning portal as a single PDF file. The portal accepts files up "
        "to twenty megabytes, and each file name should start with the student "
        "roll number followed by the course code. After the upload finishes, the "
        "portal shows a confirmation page with a receipt number, which students "
        "should save as proof of submission. Email submissions are not accepted, "
        "except when the portal is unavailable and the coordinator has announced "
        "a temporary email address in writing. The previous semester used a "
        "shared folder for submissions, and some older handouts still describe "
        "sending source files by email to the instructor; those instructions are "
        "outdated and must be ignored. Programming assignments also require the "
        "source code to be uploaded as a separate zip archive, because the PDF "
        "alone cannot be run by the marker. Resubmissions replace the earlier "
        "file only before the deadline, and the portal keeps a record of every "
        "upload time for review during any marking dispute."),
    "contractor": (
        "Remote Work Policy. This policy describes who may work away from the "
        "office and under which conditions. Employees may request remote work "
        "with manager approval, and the arrangement is reviewed every six "
        "months. Contractors may work remotely only when the signed contract "
        "explicitly permits remote work. Employee rules do not apply to "
        "contractors, so a contractor cannot rely on a manager's verbal approval "
        "or on the employee request form. Where a contract is silent on "
        "location, the contractor must work on site until the contract is "
        "amended in writing by the contracting office. All remote workers must "
        "use the company virtual private network, keep company data on approved "
        "devices, and be reachable during the core hours of ten in the morning "
        "to three in the afternoon. Interns follow a third rule: interns may "
        "work remotely for at most two days per week and only when their "
        "supervisor has confirmed that the tasks do not require on-site "
        "equipment. Any breach of the security requirements ends the remote "
        "arrangement immediately, and the case is referred to the human "
        "resources office for review."),
    "printer": (
        "Equipment Troubleshooting Notes. These notes cover common faults in the "
        "shared office devices. For blank printer pages, check ink levels and "
        "run the cleaning utility before printing a test page. Always switch the "
        "printer off and wait ten seconds before opening any cover, because the "
        "heating parts stay hot for a short time after use. If the test page is "
        "still blank after two cleaning cycles, replace the ink cartridge and "
        "record the replacement in the equipment log. Scanners have a different "
        "set of checks: for a blank scanner image, check document placement and "
        "select the correct scanning source, since a document placed face up on "
        "a glass scanner produces an empty image. Network printers also show "
        "blank pages when a wrong driver is installed, and the driver should be "
        "reinstalled from the approved software list rather than from a "
        "website. Photocopiers are covered by a separate manual and must not be "
        "opened by staff. Paper jams should be cleared slowly in the direction "
        "of the paper path to avoid tearing the sheet, and torn pieces must be "
        "removed completely before printing again."),
}

EVIDENCE = {
    "late": "Students with illness may request an extension with medical "
            "evidence before the deadline.",
    "password": "A forgotten password can be reset using the recovery link on "
                "the login page.",
    "assignment": "Upload assignments through the learning portal as a single "
                  "PDF file.",
    "contractor": "Contractors may work remotely only when the signed contract "
                  "explicitly permits remote work.",
    "printer": "For blank printer pages, check ink levels and run the cleaning "
               "utility before printing a test page.",
}


# Questions: (question, document, split). Evidence was chosen BEFORE
# looking at any search result.
QUESTIONS = [
    ("What is required for an illness extension?", "late", "dev"),
    ("How do I reset a forgotten password?", "password", "dev"),
    ("Where do I upload assignments?", "assignment", "dev"),
    ("Can contractors work remotely?", "contractor", "dev"),
    ("What are tomorrow's flight prices?", None, "dev"),
    ("What do I do for blank printer pages?", "printer", "test"),
    ("How can I recover my login access?", "password", "test"),
    ("How should I hand in my coursework?", "assignment", "test"),
    ("What proof is needed when I was ill?", "late", "test"),
    ("Who won the football match?", None, "test"),
]


def make_items():
    # Turn QUESTIONS into a list of dictionaries
    items = []
    for q in QUESTIONS:
        item = {"query": q[0], "doc": q[1], "split": q[2], "evidence": None}
        if q[1] is not None:
            item["evidence"] = EVIDENCE[q[1]]
        items.append(item)
    return items


def have_embedding():
    # Is the sentence-transformers library installed?
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


# ======================================================================
# PART 2: SEARCH CLASS (TF-IDF or embedding)
# ======================================================================

class Retriever:
    def __init__(self, records, backend="tfidf"):
        if len(records) == 0:
            raise ValueError("Give at least one record")
        self.records = records
        self.backend = backend

        self.texts = []
        for r in records:
            if r["text"].strip() == "":
                raise ValueError("A record has empty text")
            self.texts.append(r["text"])

        if backend == "tfidf":
            # TF-IDF counts words. No download needed.
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(self.texts)
        elif backend == "embedding":
            # Embedding uses a ready-made model that understands meaning.
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME, device="cpu")
            self.check_length(self.texts)
            self.matrix = self.model.encode(self.texts,
                                            normalize_embeddings=True)
        else:
            raise ValueError("backend must be tfidf or embedding")

    def check_length(self, texts):
        # The model can read only a limited number of tokens
        for text in texts:
            ids = self.model.tokenizer(text, truncation=False)["input_ids"]
            if len(ids) > self.model.max_seq_length:
                raise ValueError("Text is longer than the model limit")

    def search(self, query, k=3, filters=None):
        if query.strip() == "" or k <= 0:
            return []

        # Step 1: score every record
        if self.backend == "tfidf":
            q = self.vectorizer.transform([query])
            if q.nnz == 0:
                return []          # no common word, so no result
            scores = (q @ self.matrix.T).toarray()[0]
        else:
            self.check_length([query])
            q = self.model.encode([query], normalize_embeddings=True)[0]
            scores = self.matrix @ q

        # Step 2: keep records that match the filters
        pairs = []
        for i in range(len(self.records)):
            ok = True
            if filters is not None:
                for key in filters:
                    if self.records[i].get(key) != filters[key]:
                        ok = False
            if ok:
                pairs.append((-float(scores[i]), i))   # minus: biggest first

        # Step 3: sort and take the first k
        pairs.sort()
        results = []
        for n in range(len(pairs)):
            if n >= k:
                break
            index = pairs[n][1]
            results.append((self.records[index], float(scores[index])))
        return results


def show_hits(hits):
    if len(hits) == 0:
        print("No matching records.")
    for record, score in hits:
        print(record["id"], "score=" + format(score, ".3f"))
        print(record["text"])


# ======================================================================
# PART 3: CHUNKING (cut long text into small pieces)
# Every chunk is (start, end) = character positions. Text = text[start:end]
# ======================================================================

def word_positions(text):
    # Find start and end of every word, character by character
    positions = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i = i + 1
        else:
            start = i
            while i < n and not text[i].isspace():
                i = i + 1
            positions.append((start, i))
    return positions


def windows(text, size=40, overlap=10, offset=0):
    # Take "size" words. Next window starts (size - overlap) words later.
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Need size > 0 and 0 <= overlap < size")
    words = word_positions(text)
    result = []
    step = size - overlap
    i = 0
    while i < len(words):
        j = i + size
        if j > len(words):
            j = len(words)
        result.append((offset + words[i][0], offset + words[j - 1][1]))
        if j == len(words):
            break
        i = i + step
    return result


def sentence_spans(text):
    # Cut after . ! or ? when a space comes next
    cuts = [0]
    i = 0
    while i < len(text):
        if text[i] in ".!?" and i + 1 < len(text) and text[i + 1].isspace():
            j = i + 1
            while j < len(text) and text[j].isspace():
                j = j + 1
            cuts.append(j)
            i = j
        else:
            i = i + 1
    cuts.append(len(text))

    spans = []
    for k in range(len(cuts) - 1):
        a = cuts[k]
        b = cuts[k + 1]
        if len(text[a:b].split()) > 0:
            spans.append((a, b))
    return spans


def sentence_chunks(text, size=40):
    # Join whole sentences until the word limit. A too-long sentence
    # is cut with word windows.
    if size <= 0:
        raise ValueError("size must be positive")
    chunks = []
    has_pending = False
    pending_start = 0
    pending_end = 0
    count = 0
    for a, b in sentence_spans(text):
        n = len(text[a:b].split())
        if n > size:
            if has_pending:
                chunks.append((pending_start, pending_end))
                has_pending = False
                count = 0
            pieces = windows(text[a:b], size, 0, a)
            for p in pieces:
                chunks.append(p)
        elif has_pending and count + n <= size:
            pending_end = b
            count = count + n
        else:
            if has_pending:
                chunks.append((pending_start, pending_end))
            has_pending = True
            pending_start = a
            pending_end = b
            count = n
    if has_pending:
        chunks.append((pending_start, pending_end))
    return chunks


def make_chunks(docs, strategy):
    # strategy is "0" or "10" (overlap) or "sentence"
    result = []
    for doc_id in docs:
        text = docs[doc_id]
        if strategy == "sentence":
            spans = sentence_chunks(text)
        else:
            spans = windows(text, 40, int(strategy))
        for i in range(len(spans)):
            start = spans[i][0]
            end = spans[i][1]
            chunk = {"id": doc_id + ":" + str(i), "doc": doc_id,
                     "start": start, "end": end, "text": text[start:end]}
            result.append(chunk)
    return result


# ======================================================================
# PART 4: TASK 1 - TOKENS
# ======================================================================

def word_like(text):
    # Keep letters and digits. Everything else is a separator.
    words = []
    current = ""
    for ch in text:
        if ch.isalnum() or ch == "_":
            current = current + ch
        else:
            if current != "":
                words.append(current)
                current = ""
    if current != "":
        words.append(current)
    return words


def run_tokens():
    samples = [
        "AI, search!",                                   # punctuation
        "I can't log in.",                               # contraction
        "help@example.com",                              # email
        "CS4944",                                        # identifier
        "",                                              # empty string
        "یہ ایک مثال ہے",                                # Urdu text
        "Don't reset my password; I do NOT need it.",    # negation
        "Version 2.1 uses /v2/token/refresh",            # numbers and path
    ]

    # Optional: real model tokenizer (only if library is installed)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        print("[Model tokenizer is not available on this computer]")
        print()

    for text in samples:
        ws = text.split()
        wl = word_like(text)
        print("Original  :", repr(text))
        print("Whitespace:", ws, len(ws))
        print("Word-like :", wl, len(wl))
        print()

    if tokenizer is not None:
        print("--- Model sub-word tokens ---")
        for text in ["I can't log in.", "CS4944 help@example.com"]:
            tokens = tokenizer.tokenize(text)
            print(repr(text), tokens, len(tokens))
    else:
        print("--- Model sub-word tokens: NOT RUN (tokenizer not available) ---")

    print()
    print("Two differences:")
    print("1. split() keeps punctuation with the word (search! and can't).")
    print("   Word-like removes punctuation and breaks can't into can and t.")
    print("2. An email is 1 whitespace token but 3 word-like tokens.")
    print("Normalization risk: removing punctuation or using lowercase can")
    print("lose NOT, numbers and codes like CS4944. So keep the original text.")
    print("Note: this word-like method is not a full tokenizer for all languages.")


# ======================================================================
# PART 5: TASK 1 - FAQ CHATBOT
# It answers only when the question is exactly the same as a stored one.
# Otherwise it shows suggestions and the user must confirm.
# ======================================================================

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    # Two new FAQs I added (made-up), each with a source
    {"id": "faq:wifi", "question": "How do I connect to campus WiFi?",
     "answer": "Select the campus network and sign in with your student ID.",
     "source": "it_help.txt"},
    {"id": "faq:fees", "question": "When is the fee payment deadline?",
     "answer": "Fees are due by the date printed on the fee voucher.",
     "source": "fees.txt"}]


def build_bot(backend="tfidf"):
    records = []
    for r in FAQ:
        record = {"id": r["id"], "question": r["question"],
                  "text": r["question"] + " " + r["answer"]}
        records.append(record)
    return Retriever(records, backend)


def reply(engine, query):
    if query.strip() == "":
        return "Please enter a question."
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if len(hits) == 0:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + " (similarity " + format(score, ".3f") + ")")
    return "\n".join(lines)


def run_faq_demo():
    engine = build_bot("tfidf")
    for q in ["How do I reset my password?", "password recovery",
              "How do I connect to campus WiFi?", "", "volcano geology"]:
        print("You:", repr(q))
        print("Bot:", reply(engine, q))
        print()


def run_faq_chat():
    backend = "tfidf"
    if len(sys.argv) > 2 and sys.argv[2] == "embedding":
        backend = "embedding"
    engine = build_bot(backend)
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))


# ======================================================================
# PART 6: THE 5 SCENARIOS
# ======================================================================

def run_scenarios():
    print("===== Scenario 1: University handbook =====")
    records = [
        {"id": "assessment:late", "source": "handbook.txt",
         "text": "Late submission is normally penalized. Students with illness "
                 "may request an extension with medical evidence before the deadline."},
        {"id": "assessment:format", "source": "handbook.txt",
         "text": "Submit the assignment as a PDF through the learning portal."}]
    engine = Retriever(records)
    show_hits(engine.search("illness medical evidence late submission", k=1))
    print("-- experiment: paraphrase --")
    show_hits(engine.search("I was sick; can I hand it in tomorrow?", k=2))

    print()
    print("===== Scenario 2: Customer support =====")
    records = [
        {"id": "returns:damage", "region": "PK",
         "text": "For a damaged parcel, contact support within 48 hours "
                 "and attach photographs of the item and packaging."},
        {"id": "returns:change", "region": "PK",
         "text": "For a change of mind, return unused items within 14 days."},
        {"id": "returns:damage:other", "region": "OTHER",
         "text": "Damaged parcel reports in the other region have a 72 hour window."}]
    engine = Retriever(records)
    show_hits(engine.search("damaged parcel photographs", k=1,
                            filters={"region": "PK"}))
    print("-- experiment: no filter --")
    show_hits(engine.search("damaged parcel photographs", k=3))
    print("-- experiment: My parcel arrived broken --")
    show_hits(engine.search("My parcel arrived broken", k=3))

    print()
    print("===== Scenario 3: Software documentation =====")
    records = [
        {"id": "auth:v2", "version": "v2",
         "text": "Renew an expired API token with POST /v2/token/refresh. "
                 "Send the refresh_token field; never print its value in logs."},
        {"id": "auth:v1", "version": "v1",
         "text": "Renew an expired API token with POST /v1/auth/renew."}]
    engine = Retriever(records)
    show_hits(engine.search("renew expired API token", k=1,
                            filters={"version": "v2"}))
    print("-- experiment: v1 filter --")
    show_hits(engine.search("renew expired API token", k=1,
                            filters={"version": "v1"}))
    print("-- experiment: version v9 does not exist --")
    show_hits(engine.search("renew expired API token", k=1,
                            filters={"version": "v9"}))

    print()
    print("===== Scenario 4: Workplace policy =====")
    records = [
        {"id": "remote:staff", "audience": "employee",
         "text": "Employees may request remote work with manager approval."},
        {"id": "remote:contractor", "audience": "contractor",
         "text": "Contractors may work remotely only when the signed contract "
                 "explicitly permits remote work."}]
    engine = Retriever(records)
    show_hits(engine.search("remote work", k=1,
                            filters={"audience": "contractor"}))
    print("-- experiment: no filter --")
    show_hits(engine.search("remote work", k=2))

    print()
    print("===== Scenario 5: Equipment troubleshooting =====")
    records = [
        {"id": "printer:blank", "device": "printer",
         "text": "For blank printer pages: 1. Check ink level. "
                 "2. Run the printer cleaning utility. "
                 "3. Print a test page. Follow the device manual."},
        {"id": "scanner:blank", "device": "scanner",
         "text": "For a blank scanner image, check document placement "
                 "and select the correct scanning source."}]
    engine = Retriever(records)
    show_hits(engine.search("blank pages", k=1, filters={"device": "printer"}))
    print("-- experiment: no filter, blank image and blank pages --")
    show_hits(engine.search("blank image", k=2))
    show_hits(engine.search("blank pages", k=2))


# ======================================================================
# PART 7: METRICS (Hit@3 and MRR@3)
# Hit@3: correct result in first 3?  MRR@3: average of 1/rank (0 if none)
# ======================================================================

def retrieval_metrics(rankings, relevant):
    total_hits = 0
    total_rr = 0.0
    for n in range(len(rankings)):
        ids = rankings[n]
        good = relevant[n]
        found_rank = 0
        for r in range(3):
            if r < len(ids) and ids[r] in good:
                found_rank = r + 1
                break
        if found_rank > 0:
            total_hits = total_hits + 1
            total_rr = total_rr + 1.0 / found_rank
    count = len(rankings)
    return total_hits / count, total_rr / count


def run_metrics():
    rankings = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]]
    relevant = [{"a"}, {"a"}, {"a"}]
    hit, mrr = retrieval_metrics(rankings, relevant)
    print("Hit@3 =", round(hit, 3), "; MRR@3 =", round(mrr, 3))
    assert abs(hit - 2 / 3) < 0.000001
    assert abs(mrr - 0.5) < 0.000001
    h4, m4 = retrieval_metrics([["x", "y", "z", "a"]], [{"a"}])
    print("Rank 4 case: Hit@3 =", h4, "; MRR@3 =", m4)
    assert h4 == 0 and m4 == 0
    print("Metric tests passed")


# ======================================================================
# PART 8: TASK 2 - CHUNK TESTS AND COMPARISON
# ======================================================================

# A rule and its exception/warning. They should stay in ONE chunk.
RULES = {
    "late": "late submission is normally penalized by ten percent for each "
            "day after the deadline. Students with illness may request an "
            "extension with medical evidence before the deadline.",
    "contractor": "Contractors may work remotely only when the signed contract "
                  "explicitly permits remote work. Employee rules do not "
                  "apply to contractors,",
    "printer": "For blank printer pages, check ink levels and run the "
               "cleaning utility before printing a test page. Always switch "
               "the printer off and wait ten seconds before opening any cover,",
    "password": "A forgotten password can be reset using the recovery link on "
                "the login page. The link sends a one-time code to the "
                "registered mobile number,",
    "assignment": "Upload assignments through the learning portal as a single "
                  "PDF file. The portal accepts files up to twenty megabytes,",
}


def run_chunks():
    docs = DOCS
    for d in RULES:
        assert RULES[d] in docs[d]

    # ---- boundary tests ----
    print("--- Boundary tests ---")
    assert windows("") == []
    assert windows("short text") == [(0, 10)]
    assert sentence_chunks("") == []
    print("empty text gives no chunks: OK")

    short = "A short document."
    assert windows(short) == [(0, len(short))]
    assert sentence_chunks(short) == [(0, len(short))]
    print("text shorter than the limit gives one chunk: OK")

    for size, overlap in [(40, 40), (40, -1), (0, 0)]:
        got_error = False
        try:
            windows("a b c", size, overlap)
        except ValueError:
            got_error = True
        assert got_error
    print("bad size or overlap gives an error: OK")

    long_text = " ".join(["word"] * 95) + "."
    parts = sentence_chunks(long_text)
    sizes = []
    for a, b in parts:
        sizes.append(len(long_text[a:b].split()))
    assert len(parts) == 3
    print("sentence longer than the limit (95 words) is cut into", sizes)

    abbr = "Dr. Khan approved it. Mr. Ali did not."
    pieces = []
    for a, b in sentence_chunks(abbr, 3):
        pieces.append(abbr[a:b])
    print("abbreviation problem:", pieces)

    # ---- compare the 3 strategies ----
    original_words = 0
    for d in docs:
        original_words = original_words + len(docs[d].split())
    print()
    print("Words in each document:",
          {d: len(docs[d].split()) for d in docs})
    print("Total original words:", original_words)

    for d in docs:
        longest = 0
        for a, b in sentence_spans(docs[d]):
            n = len(docs[d][a:b].split())
            if n > longest:
                longest = n
        print("longest sentence in", d, "=", longest, "words")

    summary = {}
    names = [("window overlap 0", "0"), ("window overlap 10", "10"),
             ("sentence-aware", "sentence")]
    for label, strategy in names:
        chunks = make_chunks(docs, strategy)
        sizes = []
        for c in chunks:
            assert c["text"] == docs[c["doc"]][c["start"]:c["end"]]
            assert c["text"].strip() != ""
            sizes.append(len(c["text"].split()))
        total_emitted = sum(sizes)
        mean_size = round(total_emitted / len(sizes), 1)

        survive = {}
        intact = 0
        for d in RULES:
            found = False
            for c in chunks:
                if c["doc"] == d and RULES[d] in c["text"]:
                    found = True
            survive[d] = found
            if found:
                intact = intact + 1

        summary[label] = {"chunks": len(chunks), "min": min(sizes),
                          "max": max(sizes), "mean": mean_size,
                          "duplicate_words": total_emitted - original_words,
                          "rules_intact": intact, "survive": survive}
        print()
        print("[" + label + "] chunks =", len(chunks),
              "| words min/mean/max =", min(sizes), mean_size, max(sizes),
              "| duplicate words =", total_emitted - original_words)
        print("  rule + exception kept together:", survive)

    f = open("task2_results.json", "w", encoding="utf-8")
    json.dump(summary, f, indent=2)
    f.close()

    # ---- paraphrase queries ----
    print()
    print("--- Paraphrase queries (sentence chunks) ---")
    queries = ["How can I regain access to my account?",
               "What do I need if I was sick and missed the deadline?",
               "Where do I hand in my coursework?"]
    records = make_chunks(docs, "sentence")
    backends = ["tfidf"]
    if have_embedding():
        backends.append("embedding")
    else:
        print("[Embedding NOT RUN: sentence-transformers is not installed]")
    for backend in backends:
        engine = Retriever(records, backend)
        print()
        print("Backend:", backend)
        for q in queries:
            print(" Query:", q)
            hits = engine.search(q, k=3)
            if len(hits) == 0:
                print("   No matching records.")
            for r, s in hits:
                print("  ", r["id"], format(s, ".3f"),
                      "[" + str(r["start"]) + ":" + str(r["end"]) + "]",
                      r["text"][:80] + "...")


# ======================================================================
# PART 9: TASK 3 - FULL EVALUATION
# ======================================================================

def load_model():
    # Try to load the embedding model and measure the time
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None, "sentence-transformers is not installed"
    start = time.perf_counter()
    try:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as e:
        return None, None, "model load failed: " + str(e)
    return model, time.perf_counter() - start, None


def top3(row):
    # Numbers of the 3 chunks with the biggest scores
    pairs = []
    for i in range(len(row)):
        pairs.append((-float(row[i]), i))
    pairs.sort()
    result = []
    for k in range(3):
        if k < len(pairs):
            result.append(pairs[k][1])
    return result


def is_good(chunk, q):
    # Correct = right document AND the chunk has the full evidence
    return chunk["doc"] == q["doc"] and q["evidence"] in chunk["text"]


def evaluate(scores, chunks, items, split):
    hits = 0
    rr_sum = 0.0
    count = 0
    for n in range(len(items)):
        q = items[n]
        if q["split"] != split or q["doc"] is None:
            continue
        count = count + 1
        order = top3(scores[n])
        for r in range(len(order)):
            if is_good(chunks[order[r]], q):
                hits = hits + 1
                rr_sum = rr_sum + 1.0 / (r + 1)
                break
    if count == 0:
        return None, None
    return hits / count, rr_sum / count


def calibrate_threshold(scores, items):
    # Use only DEV questions. Threshold goes between the highest score of
    # an unanswerable question and the lowest score of an answerable one.
    best_unanswerable = -1.0
    lowest_answerable = 100.0
    have_unans = False
    have_ans = False
    for n in range(len(items)):
        q = items[n]
        if q["split"] != "dev":
            continue
        top_score = float(max(scores[n]))
        if q["doc"] is None:
            have_unans = True
            if top_score > best_unanswerable:
                best_unanswerable = top_score
        else:
            have_ans = True
            if top_score < lowest_answerable:
                lowest_answerable = top_score
    if not have_unans or not have_ans:
        return None
    if best_unanswerable >= lowest_answerable:
        return None
    return (best_unanswerable + lowest_answerable) / 2


def run_retrieval():
    docs = DOCS
    items = make_items()

    n_answerable = 0
    for q in items:
        if q["doc"] is not None:
            assert q["evidence"] in docs[q["doc"]]
            n_answerable = n_answerable + 1
    n_unanswerable = len(items) - n_answerable
    print(len(docs), "documents;", n_answerable, "answerable +",
          n_unanswerable, "unanswerable queries")

    model, load_seconds, model_error = load_model()
    if model is None:
        print("[Embedding NOT RUN]", model_error)
    else:
        print("Model load seconds:", round(load_seconds, 2))

    query_texts = []
    for q in items:
        query_texts.append(q["query"])
    original_words = 0
    for d in docs:
        original_words = original_words + len(docs[d].split())

    logs = []
    results = []
    failures = []
    token_audit = {}

    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        texts = []
        total_words = 0
        for c in chunks:
            texts.append(c["text"])
            total_words = total_words + len(c["text"].split())

        # evidence coverage: is the full evidence inside ONE chunk?
        covered = 0
        for q in items:
            if q["doc"] is None:
                continue
            for c in chunks:
                if c["doc"] == q["doc"] and q["evidence"] in c["text"]:
                    covered = covered + 1
                    break
        coverage = covered / n_answerable

        # check the model token limit
        if model is not None:
            biggest = 0
            for t in texts + query_texts:
                ids = model.tokenizer(t, truncation=False)["input_ids"]
                if len(ids) > biggest:
                    biggest = len(ids)
            token_audit[strategy] = {"max_tokens": biggest,
                                     "limit": model.max_seq_length}
            if biggest > model.max_seq_length:
                raise ValueError("Text is too long for the model")
            print("[token check] strategy", strategy, "max tokens", biggest,
                  "limit", model.max_seq_length)

        methods = ["tfidf"]
        if model is not None:
            methods.append("embedding")

        for method in methods:
            # step 1: turn texts into numbers (and time it)
            start = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
            else:
                C = model.encode(texts, normalize_embeddings=True)
                Q = model.encode(query_texts, normalize_embeddings=True)
            encode_seconds = time.perf_counter() - start

            # step 2: score every query against every chunk (and time it)
            start = time.perf_counter()
            if method == "tfidf":
                scores = (Q @ C.T).toarray()
            else:
                scores = Q @ C.T
            search_seconds = time.perf_counter() - start

            entry = {"strategy": strategy, "chunks": len(chunks),
                     "duplicate_words": total_words - original_words,
                     "evidence_coverage": coverage, "method": method,
                     "encode_seconds": encode_seconds,
                     "search_seconds": search_seconds}
            for split in ["dev", "test"]:
                h, m = evaluate(scores, chunks, items, split)
                entry[split + "_hit3"] = h
                entry[split + "_mrr3"] = m

            threshold = calibrate_threshold(scores, items)
            entry["threshold_dev"] = threshold
            if threshold is not None:
                accepted = 0
                total_ans = 0
                rejected = 0
                total_unans = 0
                for n in range(len(items)):
                    if items[n]["split"] != "test":
                        continue
                    top_score = float(max(scores[n]))
                    if items[n]["doc"] is None:
                        total_unans = total_unans + 1
                        if top_score < threshold:
                            rejected = rejected + 1
                    else:
                        total_ans = total_ans + 1
                        if top_score >= threshold:
                            accepted = accepted + 1
                entry["test_answerable_accepted"] = str(accepted) + "/" + str(total_ans)
                entry["test_unanswerable_rejected"] = str(rejected) + "/" + str(total_unans)
            results.append(entry)

            print(strategy, method, "chunks=" + str(len(chunks)),
                  "dev=" + str(entry["dev_hit3"]) + "/" + str(entry["dev_mrr3"]),
                  "test=" + str(entry["test_hit3"]) + "/" + str(entry["test_mrr3"]),
                  "encode=" + format(encode_seconds, ".4f") + "s",
                  "search=" + format(search_seconds, ".5f") + "s",
                  "threshold=" + str(threshold))

            # step 3: save top 3 for every question and find failures
            for n in range(len(items)):
                q = items[n]
                order = top3(scores[n])
                found = False
                for r in range(len(order)):
                    c = chunks[order[r]]
                    log = {"strategy": strategy, "method": method,
                           "query": q["query"], "split": q["split"],
                           "answerable": q["doc"] is not None,
                           "rank": r + 1,
                           "score": round(float(scores[n][order[r]]), 4),
                           "chunk_id": c["id"], "doc": c["doc"],
                           "start": c["start"], "end": c["end"],
                           "text": c["text"]}
                    logs.append(log)
                    if q["doc"] is not None and is_good(c, q):
                        found = True

                best = chunks[order[0]]
                best_score = round(float(scores[n][order[0]]), 4)
                if q["doc"] is not None and not found:
                    in_one_chunk = False
                    for c in chunks:
                        if c["doc"] == q["doc"] and q["evidence"] in c["text"]:
                            in_one_chunk = True
                    failures.append({
                        "strategy": strategy, "method": method,
                        "query": q["query"], "expected_doc": q["doc"],
                        "top1_chunk": best["id"], "top1_score": best_score,
                        "top1_text": best["text"],
                        "evidence_split_across_chunks": not in_one_chunk})
                if q["doc"] is None:
                    failures.append({
                        "strategy": strategy, "method": method,
                        "query": q["query"], "expected_doc": None,
                        "kind": "unanswerable still returns text",
                        "top1_chunk": best["id"], "top1_score": best_score,
                        "top1_text": best["text"]})

    # write the CSV log
    f = open("retrieval_log.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
    writer.writeheader()
    for row in logs:
        writer.writerow(row)
    f.close()

    data = {"model": MODEL_NAME, "model_load_seconds": load_seconds,
            "embedding_run": model is not None,
            "embedding_error": model_error, "token_audit": token_audit,
            "results": results, "failures": failures,
            "n_docs": len(docs), "n_answerable": n_answerable,
            "n_unanswerable": n_unanswerable}
    f = open("results.json", "w", encoding="utf-8")
    json.dump(data, f, indent=2)
    f.close()
    print("Wrote retrieval_log.csv and results.json")


# ======================================================================
# PART 10: TASK 3 - DOCUMENT CHATBOT
# Shows the best 3 passages. YOU choose one to read.
# It never says "this is the answer" by itself.
# ======================================================================

def run_document_chat():
    backend = "tfidf"
    if len(sys.argv) > 2 and sys.argv[2] == "embedding":
        backend = "embedding"
    chunks = make_chunks(DOCS, "sentence")
    try:
        engine = Retriever(chunks, backend)
    except ImportError:
        print("[sentence-transformers is not installed, using TF-IDF]")
        backend = "tfidf"
        engine = Retriever(chunks, backend)

    print("Evidence assistant (" + backend + "). Type exit to stop.")
    while True:
        try:
            query = input("Question: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break

        hits = engine.search(query)
        if len(hits) == 0:
            print("No candidate evidence; rephrase or contact a human.")
            continue

        print("Candidate evidence; relevance must be checked:")
        number = 1
        for r, score in hits:
            print(number, r["id"], format(score, ".3f"),
                  "characters", r["start"], r["end"])
            number = number + 1

        choice = input("Choose 1-3 to inspect; Enter if none apply: ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(hits):
            print("No answer selected. Rephrase or contact a human.")
            continue

        r, score = hits[int(choice) - 1]
        print("Retrieved passage:", r["text"])
        print("Source:", r["doc"], "characters", r["start"], r["end"])


# ======================================================================
# MAIN: choose what to run from the command line
# ======================================================================

def main():
    if len(sys.argv) < 2:
        print("Choose one: tokens, faq, scenarios, metrics, chunks, "
              "retrieval, doc, all")
        print("Example: python lab6_all.py retrieval")
        return
    what = sys.argv[1]
    if what == "tokens":
        run_tokens()
    elif what == "faq":
        run_faq_chat()
    elif what == "scenarios":
        run_scenarios()
    elif what == "metrics":
        run_metrics()
    elif what == "chunks":
        run_chunks()
    elif what == "retrieval":
        run_retrieval()
    elif what == "doc":
        run_document_chat()
    elif what == "all":
        print("######## TOKENS ########")
        run_tokens()
        print("######## FAQ DEMO ########")
        run_faq_demo()
        print("######## SCENARIOS ########")
        run_scenarios()
        print("######## METRICS ########")
        run_metrics()
        print("######## CHUNKS ########")
        run_chunks()
        print("######## RETRIEVAL ########")
        run_retrieval()
    else:
        print("Unknown option:", what)


if __name__ == "__main__":
    main()
