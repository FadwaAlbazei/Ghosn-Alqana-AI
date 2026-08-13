"""
غصن القنا AI — بناء السياق (Context Builder)
====================================================
المهمة: "Connect the knowledge base to the Agriculture Agent"
        (Issue #7 — Implement Memory and Knowledge Base)

هذه هي الوحدة اللي تربط كل شيء ببعض:
    1) قاعدة المعرفة (knowledge_base.py + retriever.py)
    2) تاريخ النخلة (palm_memory.py)

وتطلع "سياق" واحد جاهز (context) يُحقن مباشرة داخل prompt الـ
Agriculture Agent، بنفس الشكل المتوقع في prompts/agriculture_agent_prompt.txt
(حقل {{retrieved_knowledge_snippets}} وحقل history).

الاستخدام:
    from memory.context_builder import build_agriculture_agent_context

    context = build_agriculture_agent_context(
        palm_id="P-014",
        vision_output={"fruit_density": "high", "growth_stage": "kimri", ...},
        sensor_data={"temperature_c": 41, "humidity": 18, "soil_moisture": "low"},
    )
    # context["retrieved_knowledge_snippets"] -> يُحقن في CONTEXT (RAG)
    # context["history"]                      -> يُحقن في USER > history
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from memory.palm_memory import PalmMemory
from memory.retriever import KnowledgeRetriever


def _build_retrieval_query(vision_output: Dict[str, Any], sensor_data: Optional[Dict[str, Any]]) -> str:
    """تحويل مخرجات Vision Agent وبيانات الحساسات إلى جملة استعلام
    نصية بسيطة، تُستخدم للبحث داخل قاعدة المعرفة."""
    parts = []
    for key in ("fruit_density", "growth_stage", "bunch_condition", "color_stage", "relative_fruit_size"):
        value = vision_output.get(key)
        if value:
            parts.append(str(value))

    if sensor_data:
        for key, value in sensor_data.items():
            parts.append(f"{key} {value}")

    return " ".join(parts) if parts else "general thinning decision factors"


class AgricultureContextBuilder:
    """الواجهة الرئيسية لبناء السياق الكامل اللي يحتاجه Agriculture Agent
    قبل كل تحليل جديد. تجمع بين الذاكرة (تاريخ النخلة) والاسترجاع
    (قاعدة المعرفة عبر RAG)."""

    def __init__(
        self,
        memory: Optional[PalmMemory] = None,
        retriever: Optional[KnowledgeRetriever] = None,
    ):
        # يمكن حقن memory/retriever من الخارج (مفيد للاختبارات)،
        # أو تركهم فارغين ليتم إنشاؤهم تلقائياً بالإعدادات الافتراضية.
        self.memory = memory or PalmMemory()
        self.retriever = retriever or KnowledgeRetriever()

    def build(
        self,
        palm_id: str,
        vision_output: Dict[str, Any],
        sensor_data: Optional[Dict[str, Any]] = None,
        top_k_knowledge: int = 3,
    ) -> Dict[str, Any]:
        """بناء السياق الكامل الجاهز للحقن في prompt الـ Agriculture Agent.

        يرجع قاموس فيه:
        - retrieved_knowledge_snippets : نص جاهز (من قاعدة المعرفة عبر RAG)
        - history                      : تاريخ النخلة المختصر (من الذاكرة)
        - retrieved_chunks             : تفاصيل المقاطع المسترجعة (للتدقيق/العرض)
        """
        # 1) التأكد من وجود سجل للنخلة (ينشئه تلقائياً لو أول مرة)
        self.memory.create_palm(palm_id)

        # 2) استرجاع المعرفة الزراعية ذات الصلة بحالة النخلة الحالية
        query = _build_retrieval_query(vision_output, sensor_data)
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k_knowledge)
        knowledge_text = self.retriever.retrieve_as_context_text(query, top_k=top_k_knowledge)

        # 3) جلب تاريخ النخلة المختصر (آخر تحليلات/توصيات + قرار المزارع)
        history = self.memory.get_recent_context(palm_id)

        return {
            "palm_id": palm_id,
            "retrieved_knowledge_snippets": knowledge_text,
            "retrieved_chunks": retrieved_chunks,
            "history": history,
            "sensors": sensor_data or {},
            "vision_output": vision_output,
        }

    def build_and_record(
        self,
        palm_id: str,
        vision_output: Dict[str, Any],
        sensor_data: Optional[Dict[str, Any]] = None,
        top_k_knowledge: int = 3,
    ) -> Dict[str, Any]:
        """نفس build()، بالإضافة إلى تسجيل نتيجة Vision Agent الحالية
        كـ "تحليل جديد" داخل ذاكرة النخلة تلقائياً (previous_analysis)."""
        context = self.build(palm_id, vision_output, sensor_data, top_k_knowledge)
        self.memory.add_analysis(palm_id, vision_output)
        return context


# دالة مختصرة (shortcut) لسهولة الاستخدام المباشر بدون إنشاء الكلاس يدوياً
def build_agriculture_agent_context(
    palm_id: str,
    vision_output: Dict[str, Any],
    sensor_data: Optional[Dict[str, Any]] = None,
    top_k_knowledge: int = 3,
) -> Dict[str, Any]:
    builder = AgricultureContextBuilder()
    return builder.build_and_record(palm_id, vision_output, sensor_data, top_k_knowledge)


if __name__ == "__main__":
    import json

    # مثال كامل يحاكي تدفق البيانات الحقيقي بنفس أمثلة التوثيق (Palm P-014)
    vision_output_demo = {
        "image_quality": "good",
        "bunch_visible": True,
        "fruit_density": "high",
        "estimated_strands": 42,
        "relative_fruit_size": "small",
        "color_stage": "kimri",
        "bunch_condition": "crowded",
        "vision_confidence": 0.86,
    }
    sensor_data_demo = {"temperature_c": 41, "humidity": 18, "soil_moisture": "low"}

    context = build_agriculture_agent_context(
        palm_id="P-014",
        vision_output=vision_output_demo,
        sensor_data=sensor_data_demo,
    )

    print("=== السياق الجاهز للحقن في Agriculture Agent Prompt ===\n")
    print("--- retrieved_knowledge_snippets (مختصر) ---")
    print(context["retrieved_knowledge_snippets"][:300], "...\n")
    print("--- history ---")
    print(json.dumps(context["history"], ensure_ascii=False, indent=2))
