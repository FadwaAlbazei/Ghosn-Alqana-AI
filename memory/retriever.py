"""
غصن القنا AI — وحدة استرجاع المعرفة (Retriever)
====================================================
المهمة: "Prepare the knowledge base for RAG retrieval"
        (Issue #7 — Implement Memory and Knowledge Base)

هذه الوحدة تبحث داخل مقاطع (chunks) قاعدة المعرفة اللي جهّزناها في
knowledge_base.py، وترجّع أكثر المقاطع صلة بسؤال/حالة معينة.

التنفيذ الحالي يعتمد على تسجيل تشابه بسيط قائم على الكلمات المفتاحية
(keyword overlap / TF بسيط) بدون أي مكتبات خارجية — عشان يشتغل فوراً
بدون إنترنت أو تثبيت حزم إضافية. الشكل العام للدوال (retrieve) مصمم
عمداً بحيث يسهل استبدال المحرك الداخلي لاحقاً بـ FAISS أو Chroma
(الأدوات المذكورة في توثيق المشروع) بدون ما تتغير طريقة الاستدعاء
من باقي الكود.

الاستخدام:
    from memory.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    results = retriever.retrieve("fruit density high color sensor", top_k=3)
    for r in results:
        print(r["id"], r["title"], r["score"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from memory.knowledge_base import KnowledgeChunk, load_knowledge_base

# كلمات شائعة (عربي/إنجليزي) نتجاهلها أثناء حساب التشابه لأنها
# لا تحمل معنى تمييزي بين المقاطع (stopwords مبسطة).
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
    "are", "be", "must", "may", "should", "not", "with", "as", "if",
    "this", "that", "it", "its", "by", "at", "from", "than", "into",
    "من", "في", "على", "إلى", "أن", "إذا", "هذا", "هذه", "و", "أو",
}

_WORD_PATTERN = re.compile(r"[A-Za-z\u0600-\u06FF]+")


def _tokenize(text: str) -> List[str]:
    """تقسيم النص إلى كلمات صغيرة (lowercase) مع استبعاد الكلمات الشائعة."""
    words = _WORD_PATTERN.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


@dataclass
class RetrievedChunk:
    id: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text, "score": self.score}


class KnowledgeRetriever:
    """محرك استرجاع بسيط فوق مقاطع قاعدة المعرفة، جاهز لاستبداله
    بـ FAISS/Chroma لاحقاً دون تغيير واجهة الاستخدام (retrieve)."""

    def __init__(self, chunks: List[KnowledgeChunk] | None = None):
        self.chunks = chunks if chunks is not None else load_knowledge_base()
        # تجهيز فهرس كلمات لكل مقطع مرة واحدة فقط (كأنه "تضمين" مبسط).
        self._index: Dict[str, List[str]] = {
            c.id: _tokenize(f"{c.title} {c.text}") for c in self.chunks
        }

    def _score(self, query_words: List[str], chunk_id: str) -> float:
        """حساب درجة تشابه بسيطة = عدد الكلمات المشتركة بين السؤال
        والمقطع، مقسومة على عدد كلمات السؤال (نسبة تغطية)."""
        if not query_words:
            return 0.0
        chunk_words = set(self._index[chunk_id])
        overlap = sum(1 for w in query_words if w in chunk_words)
        return overlap / len(query_words)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """إرجاع أكثر top_k مقاطع صلة بالسؤال/الحالة المُدخلة.

        query: يمكن أن يكون سؤال حر، أو ملخص حالة (مثال:
               "fruit density high, color sensor missing, unknown variety").
        """
        query_words = _tokenize(query)
        scored = [
            RetrievedChunk(
                id=c.id,
                title=c.title,
                text=c.text,
                score=self._score(query_words, c.id),
            )
            for c in self.chunks
        ]
        scored.sort(key=lambda r: r.score, reverse=True)

        # لو كل الدرجات صفر (سؤال غير مطابق لأي كلمة)، نرجع أهم قواعد
        # السلامة والمبادئ العامة كافتراضي آمن بدل ما نرجع فراغ.
        top = [r for r in scored if r.score > 0][:top_k]
        if not top:
            fallback_ids = {"kb-14", "kb-08", "kb-05"}  # السلامة + مبادئ القرار
            top = [
                RetrievedChunk(id=c.id, title=c.title, text=c.text, score=0.0)
                for c in self.chunks
                if c.id in fallback_ids
            ][:top_k]

        return [r.to_dict() for r in top]

    def retrieve_as_context_text(self, query: str, top_k: int = 3) -> str:
        """نفس retrieve() لكن يرجّع النتائج كنص واحد جاهز للحقن المباشر
        داخل {{retrieved_knowledge_snippets}} في prompt الـ Agriculture Agent."""
        results = self.retrieve(query, top_k=top_k)
        blocks = [f"[{r['id']}] {r['title']}\n{r['text']}" for r in results]
        return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    retriever = KnowledgeRetriever()

    # مثال: حالة فيها كثافة ثمار عالية وقراءة حساس لون غير مؤكدة
    demo_query = "high fruit density color sensor unreliable kimri growth stage"
    print("=== نتائج الاسترجاع لحالة تجريبية ===")
    for r in retriever.retrieve(demo_query, top_k=3):
        print(f"- {r['id']} | {r['title']} | score={r['score']:.2f}")

    print("\n=== نفس النتائج كنص جاهز للـ prompt ===")
    print(retriever.retrieve_as_context_text(demo_query, top_k=2))
