"""
RAG (Retrieval-Augmented Generation) Engine.
Provides semantic search and document retrieval over company support docs,
technical troubleshooting guides, and product catalogs.
"""

import math
import re
from typing import List, Dict, Any, Tuple

# Enterprise Support Knowledge Base
SUPPORT_KNOWLEDGE_DOCS = [
    {
        "id": "kb_bill_001",
        "category": "billing",
        "title": "Payment Deducted Without Order Confirmation",
        "content": (
            "When a customer has funds deducted from their account but receives no order confirmation, "
            "it is typically due to an asynchronous webhook timeout between the merchant payment gateway "
            "(e.g., Stripe, PayPal) and our order orchestration system. "
            "Resolution: 1. Verify transaction status via Payment MCP tool. "
            "2. If status is 'CHARGED' and order is 'PENDING', trigger order replay. "
            "3. If order replay fails after 30 minutes, bank authorization is automatically voided, "
            "releasing hold back to customer account within 3 to 5 business days."
        )
    },
    {
        "id": "kb_bill_002",
        "category": "billing",
        "title": "Refund and Chargeback Policy",
        "content": (
            "Customers are eligible for a 100% refund within 30 days of purchase for software licenses. "
            "Refunds exceeding $100 require Human Support Supervisor sign-off. "
            "Duplicate charges are automatically refunded upon detection."
        )
    },
    {
        "id": "kb_tech_001",
        "category": "technical",
        "title": "Resolving 504 Gateway Timeout Errors",
        "content": (
            "A 504 Gateway Timeout indicates our reverse proxy or load balancer did not receive a timely response "
            "from the upstream application server. "
            "Troubleshooting steps: 1. Check https://status.ourcloud.com for ongoing platform incidents. "
            "2. If status is operational, verify client connection and ensure request payloads do not exceed 10MB. "
            "3. Implement exponential backoff with jitter on API calls."
        )
    },
    {
        "id": "kb_tech_002",
        "category": "technical",
        "title": "API Authentication and Token Expiration",
        "content": (
            "401 Unauthorized errors occur when Bearer tokens are invalid or expired. "
            "Access tokens expire after 60 minutes. Use the refresh token endpoint (/api/v1/auth/refresh) "
            "to obtain a new JWT without prompting user re-login."
        )
    },
    {
        "id": "kb_prod_001",
        "category": "product",
        "title": "CloudSync Pro Compatibility and Offline Mode",
        "content": (
            "CloudSync Pro is fully compatible with Windows 11, macOS Sequoia, and Ubuntu 22.04 LTS. "
            "It supports local offline file editing with zero data loss. "
            "Offline changes are encrypted with AES-256 and queued until network connectivity is restored."
        )
    }
]

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) vector retrieval engine.
    Computes vector embeddings and cosine similarity ranking.
    """
    def __init__(self, documents: List[Dict[str, Any]] = SUPPORT_KNOWLEDGE_DOCS):
        self.documents = documents
        self.vocabulary: Dict[str, int] = {}
        self.doc_vectors: List[List[float]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and tokenizes text into lowercase normalized terms."""
        return re.findall(r"\b\w{3,}\b", text.lower())

    def _build_index(self):
        """Constructs vector vocabulary and TF-IDF normalized vector representation."""
        all_tokens = []
        for doc in self.documents:
            full_text = f"{doc['title']} {doc['content']} {doc['category']}"
            tokens = self._tokenize(full_text)
            all_tokens.extend(tokens)
            doc["_tokens"] = tokens

        unique_words = sorted(list(set(all_tokens)))
        self.vocabulary = {word: idx for idx, word in enumerate(unique_words)}
        
        # Calculate TF-IDF vectors
        total_docs = len(self.documents)
        df = {word: 0 for word in self.vocabulary}
        for doc in self.documents:
            unique_doc_words = set(doc["_tokens"])
            for word in unique_doc_words:
                if word in df:
                    df[word] += 1

        self.idf = {word: math.log((total_docs + 1) / (df[word] + 1)) + 1 for word in self.vocabulary}

        self.doc_vectors = []
        for doc in self.documents:
            vec = self._vectorize(doc["_tokens"])
            self.doc_vectors.append(vec)

    def _vectorize(self, tokens: List[str]) -> List[float]:
        vec = [0.0] * len(self.vocabulary)
        if not tokens:
            return vec
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        
        norm_sq = 0.0
        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                val = (count / len(tokens)) * self.idf.get(word, 1.0)
                vec[idx] = val
                norm_sq += val * val
        
        norm = math.sqrt(norm_sq)
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        return sum(a * b for a, b in zip(vec1, vec2))

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves top-k most relevant knowledge documents for a given user query."""
        query_tokens = self._tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            sim = self._cosine_similarity(query_vec, self.doc_vectors[idx])
            scored_docs.append((sim, doc))

        # Sort descending by similarity score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "similarity_score": round(score, 4)
            })
        return results

    def get_grounded_context(self, query: str, top_k: int = 2) -> str:
        """Formats retrieved documents into a clean prompt context string."""
        docs = self.retrieve(query, top_k=top_k)
        if not docs or docs[0]["similarity_score"] < 0.05:
            return "No directly matching knowledge base document found."
        
        context_parts = []
        for i, d in enumerate(docs, 1):
            context_parts.append(
                f"[Doc {i}: {d['title']} (Category: {d['category']}, Relevance: {d['similarity_score']})]\n{d['content']}"
            )
        return "\n\n".join(context_parts)

# Global RAG instance
rag_engine = RAGEngine()

if __name__ == "__main__":
    test_q = "My payment was deducted, but my order was not confirmed."
    retrieved = rag_engine.retrieve(test_q, top_k=2)
    print("--- RAG Retrieval Test ---")
    for r in retrieved:
        print(f"Title: {r['title']} | Score: {r['similarity_score']}")
        print(f"Content: {r['content'][:120]}...\n")
