import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import groq
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ----------------------------
# Source-Grounded RAG Chatbot
# ----------------------------
class SourceGroundedChatbot:
    def __init__(self):
        self.client = None
        self.model = "llama-3.1-8b-instant"
        self.documents = []  # List of dicts: {url, lines, chunks, line_ranges}
        self.vectorizer = None
        self.embeddings_matrix = None
        self.chunk_meta = []

    # Initialize Groq
    def init_groq(self, api_key):
        try:
            self.client = groq.Groq(api_key=api_key)
            # test
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            st.error(f"Groq init failed: {e}")
            return False

    # Extract all text lines from URL
    def extract_full_text(self, url: str) -> List[str]:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            lines = soup.get_text(separator="\n").split("\n")
            lines = [line.strip() for line in lines if line.strip()]
            return lines
        except Exception as e:
            st.warning(f"Failed to extract {url}: {str(e)}")
            return []

    # Chunk lines into groups, store line ranges
    def chunk_lines(self, lines: List[str], max_lines=20):
        chunks = []
        line_ranges = []
        for i in range(0, len(lines), max_lines):
            chunk = " ".join(lines[i:i+max_lines])
            chunks.append(chunk)
            line_ranges.append((i+1, min(i+max_lines, len(lines))))
        return chunks, line_ranges

    # Process multiple URLs and compute embeddings
    def process_urls(self, urls: List[str]) -> int:
        self.documents = []
        all_chunks = []
        self.chunk_meta = []
        progress = st.progress(0.0)

        for i, url in enumerate(urls):
            lines = self.extract_full_text(url)
            if not lines:
                self.documents.append({"url": url, "error": "No text extracted"})
                progress.progress((i+1)/len(urls))
                continue

            chunks, line_ranges = self.chunk_lines(lines)
            self.documents.append({
                "url": url,
                "lines": lines,
                "chunks": chunks,
                "line_ranges": line_ranges
            })

            # Store chunk metadata
            for c, lr in zip(chunks, line_ranges):
                all_chunks.append(c)
                self.chunk_meta.append({"url": url, "chunk_text": c, "line_range": lr})

            progress.progress((i+1)/len(urls))

        # Compute TF-IDF embeddings for all chunks
        if all_chunks:
            self.vectorizer = TfidfVectorizer().fit(all_chunks)
            self.embeddings_matrix = self.vectorizer.transform(all_chunks)

        st.session_state.docs = self.documents
        return len(self.documents)

    # Retrieve top-k relevant chunks with metadata
    def retrieve_relevant_chunks(self, question: str, top_k=5):
        if self.embeddings_matrix is None or self.vectorizer is None:
            return []

        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.embeddings_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant = []
        for i in top_indices:
            if similarities[i] > 0.01:  # threshold
                relevant.append(self.chunk_meta[i])
        return relevant

    # Answer question
    def answer(self, question: str):
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k=5)
        if not relevant_chunks:
            return "I cannot determine the answer from the data."

        # Build context with source info
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"Source: {chunk['url']} (lines {chunk['line_range'][0]}-{chunk['line_range'][1]})\n{chunk['chunk_text']}")

        context_text = "\n---\n".join(context_parts)

        prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
Do NOT guess. If the answer is not in context, say "I cannot determine the answer from the data."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            return res.choices[0].message.content
        except Exception as e:
            return f"AI error: {e}"


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Source-Grounded Chatbot", layout="wide")
    st.title("🧠 Source-Grounded Chatbot (Line-by-Line)")

    # Initialize session state
    if "bot" not in st.session_state:
        st.session_state.bot = SourceGroundedChatbot()
    if "docs" not in st.session_state:
        st.session_state.docs = []

    # Sidebar: API key
    with st.sidebar:
        st.header("🔑 Groq API Key")
        api_key = st.text_input("Enter your API Key", type="password")
        if st.button("Connect"):
            if st.session_state.bot.init_groq(api_key):
                st.success("Connected successfully!")

    # Step 1: URLs input
    st.subheader("🔗 Step 1: Enter URLs")
    urls_text = st.text_area(
        "Enter one URL per line",
        height=120,
        value="https://en.wikipedia.org/wiki/Elon_Musk"
    )

    if st.button("Process URLs"):
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
        count = st.session_state.bot.process_urls(url_list)
        st.success(f"Processed {count} URL(s)")

    # Step 2: Display extracted lines
    if st.session_state.docs:
        st.subheader("📄 Extracted Lines (first 50 lines per URL)")
        for doc in st.session_state.docs:
            with st.expander(doc["url"]):
                if "error" in doc:
                    st.warning(doc["error"])
                else:
                    for line in doc["lines"][:50]:
                        st.write(line)

        # Step 3: Ask questions
        st.subheader("💬 Ask Questions")
        question = st.text_input("Enter your question here")
        if st.button("Ask"):
            if question.strip():
                answer = st.session_state.bot.answer(question)
                st.info(answer)
            else:
                st.warning("Please enter a question!")

if __name__ == "__main__":
    main()
