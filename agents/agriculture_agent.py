
# ============================================================
# نظام الذكاء الاصطناعي لتقييم نخيل التمر
# ============================================================

!pip -q install sentence-transformers

import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("نظام حصين للذكاء الاصطناعي الزراعي")
print("=" * 70)
print("تم تحميل المكتبات بنجاح")


# ============================================================
# 1. مخرجات وكيل الرؤية Vision Agent
# ============================================================

vision_agent_output = {
    "palm_id": "P-102",
    "fruit_bunches": 12,
    "estimated_fruit_load": "High",
    "fruit_density": "High",
    "fruit_size_estimate": "Medium",
    "visual_issues": [
        "Dense fruit clusters"
    ],
    "confidence": 0.91
}


# ============================================================
# 2. بيانات الحساسات
# ============================================================

sensor_data = {
    "temperature_c": 43.2,
    "humidity_pct": 18,
    "soil_moisture_pct": 21,
    "soil_ec": 3.1,
    "soil_temperature_c": 31.5
}


# ============================================================
# 3. معلومات النخلة
# ============================================================

palm_information = {
    "palm_id": "P-102",
    "variety": "Khalas",
    "growth_stage": "Fruit Development",
    "age_years": 12
}


# ============================================================
# 4. تاريخ النخلة
# ============================================================

palm_history = {
    "history_available": True,
    "previous_yield_kg": 95,
    "previous_thinning": True,
    "previous_fruit_load": "High",
    "last_inspection_date": "2026-07-20"
}


# ============================================================
# 5. قاعدة المعرفة الزراعية
# ============================================================
#
# هذه بيانات تجريبية.
# لاحقًا نستبدلها بمراجع زراعية حقيقية.
# ============================================================

knowledge_base = [

    {
        "id": "KB001",
        "topic": "تخفيف الثمار",
        "text": """
        يمكن النظر في تخفيف الثمار عندما يكون الحمل الثمري مرتفعًا
        أو عندما تكون عناقيد الثمار كثيفة بشكل زائد.
        يجب أن يعتمد القرار على صنف النخلة ومرحلة النمو وكثافة الثمار
        وحالة النخلة وأهداف الإنتاج.
        """
    },

    {
        "id": "KB002",
        "topic": "المنافسة بين الثمار",
        "text": """
        ارتفاع كثافة الثمار قد يؤدي إلى زيادة المنافسة بين الثمار
        على الموارد المتاحة. يجب أخذ الحمل الثمري وتوزيع الثمار
        وحجم الثمار ومرحلة النمو وحالة النخلة في الاعتبار.
        """
    },

    {
        "id": "KB003",
        "topic": "اتخاذ القرار الزراعي",
        "text": """
        يجب ألا يعتمد القرار الزراعي على قياس واحد فقط.
        ينبغي الجمع بين ملاحظات الرؤية وبيانات الحساسات وصنف النخلة
        ومرحلة النمو والمعلومات التاريخية عند توفرها.
        """
    },

    {
        "id": "KB004",
        "topic": "تاريخ النخلة",
        "text": """
        يمكن استخدام المعلومات التاريخية للنخلة لإضافة سياق للقرار.
        يمكن الاستفادة من الإنتاج السابق والحمل الثمري السابق
        والإجراءات الزراعية السابقة عند توفرها.
        ولا يجوز اختراع معلومات تاريخية غير موجودة.
        """
    },

    {
        "id": "KB005",
        "topic": "الثقة في القرار",
        "text": """
        عندما تكون ملاحظات الرؤية غير مؤكدة أو تكون بيانات الحساسات
        ناقصة، يجب خفض مستوى الثقة والتوصية بالتحقق الميداني
        بدل تقديم القرار غير المؤكد على أنه حقيقة.
        """
    }
]


# ============================================================
# 6. تحميل نموذج RAG
# ============================================================

print("\nجاري تحميل نموذج البحث في قاعدة المعرفة...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("تم تحميل نموذج RAG بنجاح")


# ============================================================
# 7. تحويل قاعدة المعرفة إلى Embeddings
# ============================================================

knowledge_documents = [
    item["text"]
    for item in knowledge_base
]

knowledge_embeddings = embedding_model.encode(
    knowledge_documents,
    normalize_embeddings=True
)


# ============================================================
# 8. إنشاء سؤال البحث RAG
# ============================================================

rag_query = f"""
صنف النخلة: {palm_information['variety']}

مرحلة النمو: {palm_information['growth_stage']}

عدد عناقيد الثمار: {vision_agent_output['fruit_bunches']}

الحمل الثمري المتوقع: {vision_agent_output['estimated_fruit_load']}

كثافة الثمار: {vision_agent_output['fruit_density']}

حجم الثمار المتوقع: {vision_agent_output['fruit_size_estimate']}

درجة الحرارة: {sensor_data['temperature_c']} درجة مئوية

رطوبة التربة: {sensor_data['soil_moisture_pct']} %

الإنتاج السابق: {palm_history['previous_yield_kg']} كجم

السؤال:
هل ينبغي النظر في تخفيف الثمار؟
وما العوامل الزراعية التي يجب أخذها في الاعتبار؟
"""


# ============================================================
# 9. البحث عن المعلومات الزراعية المناسبة
# ============================================================

query_embedding = embedding_model.encode(
    [rag_query],
    normalize_embeddings=True
)[0]

similarities = np.dot(
    knowledge_embeddings,
    query_embedding
)

top_k = 3

top_indices = np.argsort(
    similarities
)[::-1][:top_k]

retrieved_knowledge = []

for index in top_indices:

    retrieved_knowledge.append({
        "id": knowledge_base[index]["id"],
        "topic": knowledge_base[index]["topic"],
        "text": knowledge_base[index]["text"],
        "similarity": round(
            float(similarities[index]),
            4
        )
    })


# ============================================================
# 10. تقييم الحمل الثمري
# ============================================================

def assess_fruit_load(vision_data):

    score = 0

    fruit_bunches = vision_data["fruit_bunches"]
    estimated_load = vision_data["estimated_fruit_load"]
    density = vision_data["fruit_density"]

    if fruit_bunches >= 10:
        score += 2

    if estimated_load == "High":
        score += 2

    if density == "High":
        score += 2

    if score >= 5:
        level = "مرتفع"

    elif score >= 3:
        level = "متوسط"

    else:
        level = "منخفض"

    return {
        "level": level,
        "score": score
    }


fruit_load_assessment = assess_fruit_load(
    vision_agent_output
)


# ============================================================
# 11. تقييم المنافسة بين الثمار
# ============================================================

def assess_competition(
    vision_data,
    fruit_load_assessment
):

    fruit_load = fruit_load_assessment["level"]
    density = vision_data["fruit_density"]

    if (
        fruit_load == "مرتفع"
        and density == "High"
    ):

        risk = "مرتفع"

    elif (
        fruit_load == "مرتفع"
        or density == "High"
    ):

        risk = "متوسط"

    else:

        risk = "منخفض"

    return risk


competition_risk = assess_competition(
    vision_agent_output,
    fruit_load_assessment
)


# ============================================================
# 12. تحديد هل نحتاج للنظر في التخفيف
# ============================================================

def determine_thinning(
    vision_data,
    palm_data,
    history_data,
    fruit_load,
    competition
):

    confidence = vision_data["confidence"]

    if (
        fruit_load == "مرتفع"
        and competition == "مرتفع"
    ):

        if confidence >= 0.85:

            decision = "ينبغي النظر في التخفيف"

            priority = "عالية"

            reason = (
                "تم رصد حمل ثمري مرتفع وكثافة مرتفعة للثمار، "
                "مما يشير إلى احتمال وجود منافسة بين الثمار. "
                "ينبغي التحقق ميدانيًا قبل اتخاذ الإجراء."
            )

        else:

            decision = "ينبغي النظر في التخفيف"

            priority = "متوسطة"

            reason = (
                "تشير بيانات الرؤية إلى حمل ثمري مرتفع، "
                "لكن مستوى الثقة لا يكفي لاتخاذ توصية قوية."
            )

    elif (
        fruit_load == "مرتفع"
        or competition == "مرتفع"
    ):

        decision = "ينبغي النظر في التخفيف"

        priority = "متوسطة"

        reason = (
            "هناك مؤشرات على ارتفاع الحمل الثمري أو المنافسة "
            "بين الثمار، ويوصى بإجراء تقييم ميداني إضافي."
        )

    else:

        decision = "لا يوصى بالتخفيف حاليًا"

        priority = "منخفضة"

        reason = (
            "المؤشرات الحالية لا توفر أدلة كافية على الحاجة "
            "إلى تخفيف الثمار."
        )

    return {
        "decision": decision,
        "priority": priority,
        "reason": reason
    }


thinning_result = determine_thinning(
    vision_agent_output,
    palm_information,
    palm_history,
    fruit_load_assessment["level"],
    competition_risk
)


# ============================================================
# 13. إنشاء التفسير الزراعي
# ============================================================

def generate_reasoning(
    vision_data,
    sensor_data,
    palm_data,
    history_data,
    fruit_load,
    competition
):

    reasoning = []

    reasoning.append(
        f"صنف النخلة هو {palm_data['variety']}."
    )

    reasoning.append(
        f"مرحلة النمو الحالية هي {palm_data['growth_stage']}."
    )

    reasoning.append(
        f"تم تقدير عدد عناقيد الثمار بـ "
        f"{vision_data['fruit_bunches']} عنقودًا."
    )

    reasoning.append(
        f"تم تصنيف الحمل الثمري على أنه {fruit_load}."
    )

    reasoning.append(
        f"تم تصنيف خطر المنافسة بين الثمار على أنه {competition}."
    )

    if vision_data["fruit_density"] == "High":

        reasoning.append(
            "تم رصد كثافة مرتفعة للثمار بواسطة وكيل الرؤية."
        )

    if history_data["history_available"]:

        reasoning.append(
            f"الإنتاج السابق للنخلة كان "
            f"{history_data['previous_yield_kg']} كجم."
        )

        reasoning.append(
            "تم أخذ التاريخ السابق للنخلة في الاعتبار."
        )

    else:

        reasoning.append(
            "لا تتوفر معلومات تاريخية كافية عن النخلة."
        )

    reasoning.append(
        f"درجة الحرارة الحالية "
        f"{sensor_data['temperature_c']} درجة مئوية."
    )

    reasoning.append(
        f"رطوبة التربة الحالية "
        f"{sensor_data['soil_moisture_pct']}%."
    )

    reasoning.append(
        "يجب التحقق من حالة النخلة ميدانيًا قبل اتخاذ "
        "أي إجراء زراعي فعلي."
    )

    return reasoning


agricultural_reasoning = generate_reasoning(
    vision_agent_output,
    sensor_data,
    palm_information,
    palm_history,
    fruit_load_assessment["level"],
    competition_risk
)


# ============================================================
# 14. حساب مستوى الثقة
# ============================================================

vision_confidence = vision_agent_output["confidence"]

if competition_risk == "مرتفع":

    assessment_factor = 0.90

elif competition_risk == "متوسط":

    assessment_factor = 0.80

else:

    assessment_factor = 0.75


overall_confidence = round(
    vision_confidence * assessment_factor,
    2
)


# ============================================================
# 15. النتيجة النهائية بصيغة JSON
# ============================================================

final_result = {

    "النظام": "حصين - الذكاء الاصطناعي الزراعي",

    "النخلة": {
        "رقم_النخلة": palm_information["palm_id"],
        "الصنف": palm_information["variety"],
        "مرحلة_النمو": palm_information["growth_stage"],
        "العمر_بالسنوات": palm_information["age_years"]
    },

    "وكيل_الرؤية": {
        "عدد_عناقيد_الثمار":
            vision_agent_output["fruit_bunches"],

        "الحمل_الثمري_المقدر":
            vision_agent_output["estimated_fruit_load"],

        "كثافة_الثمار":
            vision_agent_output["fruit_density"],

        "حجم_الثمار_المقدر":
            vision_agent_output["fruit_size_estimate"],

        "المشكلات_المرئية":
            vision_agent_output["visual_issues"],

        "مستوى_الثقة":
            vision_agent_output["confidence"]
    },

    "بيانات_الحساسات": sensor_data,

    "تاريخ_النخلة": palm_history,

    "التقييم": {

        "الحمل_الثمري": fruit_load_assessment,

        "خطر_المنافسة_بين_الثمار": competition_risk,

        "الثقة_العامة": overall_confidence
    },

    "توصية_التخفيف": thinning_result,

    "التفسير_الزراعي": agricultural_reasoning,

    "المعرفة_المسترجعة": [
        {
            "المعرف": item["id"],
            "الموضوع": item["topic"],
            "درجة_التشابه": item["similarity"]
        }
        for item in retrieved_knowledge
    ],

    "ملاحظة": (
        "هذا النظام نموذج أولي لدعم القرار. "
        "يجب التحقق من التوصية بواسطة مختص زراعي "
        "وبناءً على ظروف الموقع الفعلية."
    )
}


# ============================================================
# 16. طباعة النتيجة
# ============================================================

print("\n")
print("=" * 70)
print("النتيجة النهائية")
print("=" * 70)

print(
    json.dumps(
        final_result,
        ensure_ascii=False,
        indent=2
    )
)

print("=" * 70)
print("اكتمل التحليل بنجاح")
