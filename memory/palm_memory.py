"""
غصن القنا AI — وحدة ذاكرة النخلة (Palm Memory)
====================================================
المهمة: "Create palm history storage for the MVP" + كل البنود المرتبطة بها
        (Issue #7 — Implement Memory and Knowledge Base)

هذا الملف مسؤول عن تخزين واسترجاع تاريخ كل نخلة على حدة:
- معرف النخلة (Palm ID)
- التحليلات السابقة (نتائج Vision Agent و Agriculture Agent)
- التوصيات السابقة (نتائج Decision Agent)
- قرار المزارع (موافقة / تعديل / رفض)
- نتائج ما بعد التخفيف (متابعة)
- موعد الفحص القادم

بما إن المشروع (حسب توثيق الهاكاثون) خطط لاستخدام Firebase أو Supabase،
لكن ما فيه اتصال إنترنت متاح هنا للتجربة، هذا الملف يستخدم تخزين محلي
بصيغة JSON بنفس شكل البيانات (schema) المخطط له تماماً. يعني لاحقاً
لما يتوفر Firebase/Supabase، يكفي تبديل دوال القراءة/الكتابة فقط
(save/load) بدون ما نغيّر باقي منطق البرنامج.

الاستخدام:
    from memory.palm_memory import PalmMemory

    memory = PalmMemory()
    memory.create_palm("P-014", location="Block A - Row 3", variety="Sukari")
    memory.add_analysis("P-014", {"fruit_density": "high", "growth_stage": "kimri"})
    memory.add_recommendation("P-014", {"thinning_needed": True, "thinning_percentage": 25})
    memory.record_farmer_action("P-014", "accepted")
    print(memory.get_palm_history("P-014"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# مسار ملف تخزين الذاكرة (يعادل قاعدة البيانات المؤقتة بدل Firebase/Supabase)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = _REPO_ROOT / "memory" / "data" / "palm_memory.json"

# قيم مسموحة لقرار المزارع، مطابقة للمخطط في التوثيق (farmer_action enum)
FARMER_ACTIONS = {"accepted", "modified", "rejected", "pending"}


def _now_iso() -> str:
    """تاريخ ووقت الآن بصيغة ISO موحّدة تُستخدم في كل السجلات."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class PalmRecord:
    """سجل نخلة واحدة، مطابق تماماً للحقول الموضحة في Memory Design
    داخل توثيق المشروع (project-documentation.pdf)."""

    palm_id: str
    variety: Optional[str] = None
    location: Optional[str] = None          # geo / block location
    image_url: Optional[str] = None
    previous_analysis: List[Dict[str, Any]] = field(default_factory=list)
    previous_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    farmer_action: str = "pending"           # accepted / modified / rejected / pending
    results_after_thinning: List[Dict[str, Any]] = field(default_factory=list)
    next_check_date: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "PalmRecord":
        return PalmRecord(**data)


class PalmMemory:
    """الواجهة الرئيسية للتعامل مع ذاكرة/تاريخ النخيل.

    كل الدوال هنا تقرأ وتكتب في ملف JSON محلي، لكن التصميم يسمح
    باستبدال الطبقة السفلية لاحقاً بـ Firebase/Supabase بسهولة، لأن
    كل شيء يمر عبر _load() و _save() فقط.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._save({})

    # ---------- طبقة القراءة/الكتابة (يمكن استبدالها بـ Firebase لاحقاً) ----------

    def _load(self) -> Dict[str, dict]:
        if not self.db_path.exists():
            return {}
        raw = self.db_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        return json.loads(raw)

    def _save(self, data: Dict[str, dict]) -> None:
        self.db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---------- إنشاء نخلة جديدة وتخزين Palm ID ----------

    def create_palm(
        self,
        palm_id: str,
        variety: Optional[str] = None,
        location: Optional[str] = None,
    ) -> PalmRecord:
        """تسجيل نخلة جديدة بمعرفها الفريد (Palm ID). إذا كانت موجودة
        مسبقاً، ترجع السجل الحالي بدون تكرار."""
        data = self._load()
        if palm_id in data:
            return PalmRecord.from_dict(data[palm_id])

        record = PalmRecord(palm_id=palm_id, variety=variety, location=location)
        data[palm_id] = record.to_dict()
        self._save(data)
        return record

    # ---------- تخزين التحليل السابق ----------

    def add_analysis(self, palm_id: str, analysis: Dict[str, Any]) -> None:
        """إضافة نتيجة تحليل جديدة (من Vision Agent / Agriculture Agent)
        إلى سجل previous_analysis الخاص بالنخلة."""
        data = self._load()
        record = data.setdefault(palm_id, PalmRecord(palm_id=palm_id).to_dict())

        entry = dict(analysis)
        entry["timestamp"] = _now_iso()
        record["previous_analysis"].append(entry)

        # آخر صورة مرفقة بالتحليل، إن وجدت، تُحدّث كصورة مرجعية للنخلة
        if "image_url" in analysis:
            record["image_url"] = analysis["image_url"]

        record["updated_at"] = _now_iso()
        data[palm_id] = record
        self._save(data)

    # ---------- تخزين التوصية السابقة ----------

    def add_recommendation(self, palm_id: str, recommendation: Dict[str, Any]) -> None:
        """إضافة توصية جديدة (من Decision Agent) إلى سجل
        previous_recommendations الخاص بالنخلة."""
        data = self._load()
        record = data.setdefault(palm_id, PalmRecord(palm_id=palm_id).to_dict())

        entry = dict(recommendation)
        entry["timestamp"] = _now_iso()
        record["previous_recommendations"].append(entry)

        # كل توصية جديدة تعيد حالة قرار المزارع إلى "بانتظار الرد"
        record["farmer_action"] = "pending"

        # تحديد موعد الفحص القادم تلقائياً إن كان متوفر ضمن التوصية
        next_days = recommendation.get("next_check_days")
        if next_days:
            next_date = datetime.now(timezone.utc) + timedelta(days=int(next_days))
            record["next_check_date"] = next_date.date().isoformat()

        record["updated_at"] = _now_iso()
        data[palm_id] = record
        self._save(data)

    # ---------- تخزين موافقة/رفض المزارع ----------

    def record_farmer_action(
        self,
        palm_id: str,
        action: str,
        note: Optional[str] = None,
    ) -> None:
        """تسجيل قرار المزارع تجاه آخر توصية: accepted / modified / rejected."""
        if action not in FARMER_ACTIONS:
            raise ValueError(
                f"قيمة غير صحيحة لقرار المزارع: '{action}'. "
                f"القيم المسموحة: {sorted(FARMER_ACTIONS)}"
            )

        data = self._load()
        if palm_id not in data:
            raise KeyError(f"لا يوجد سجل للنخلة '{palm_id}'. أنشئيها أولاً بـ create_palm().")

        record = data[palm_id]
        record["farmer_action"] = action
        if note:
            record.setdefault("farmer_notes", []).append(
                {"note": note, "timestamp": _now_iso()}
            )
        record["updated_at"] = _now_iso()
        data[palm_id] = record
        self._save(data)

    # ---------- تخزين نتائج ما بعد التخفيف (متابعة) ----------

    def add_result_after_thinning(self, palm_id: str, result: Dict[str, Any]) -> None:
        """تسجيل نتيجة المتابعة بعد تنفيذ التخفيف فعلياً (مثال: تحسّن
        كثافة الثمار، حجم الثمرة بعد فترة)."""
        data = self._load()
        if palm_id not in data:
            raise KeyError(f"لا يوجد سجل للنخلة '{palm_id}'. أنشئيها أولاً بـ create_palm().")

        entry = dict(result)
        entry["timestamp"] = _now_iso()
        data[palm_id]["results_after_thinning"].append(entry)
        data[palm_id]["updated_at"] = _now_iso()
        self._save(data)

    # ---------- استرجاع تاريخ نخلة أثناء التحليل ----------

    def get_palm_history(self, palm_id: str) -> Optional[Dict[str, Any]]:
        """استرجاع السجل الكامل لنخلة معينة. هذه الدالة هي المستخدَمة
        من قبل Agriculture Agent أثناء تحليل صورة جديدة، لمعرفة
        الوضع التاريخي للنخلة قبل إصدار توصية جديدة."""
        data = self._load()
        return data.get(palm_id)

    def get_recent_context(self, palm_id: str, max_items: int = 3) -> Dict[str, Any]:
        """نسخة مختصرة من تاريخ النخلة (آخر عدد من التحليلات والتوصيات
        فقط)، مناسبة للحقن المباشر داخل الـ prompt بدون إثقاله بسجل طويل."""
        history = self.get_palm_history(palm_id)
        if not history:
            return {
                "palm_id": palm_id,
                "has_history": False,
                "note": "لا يوجد تاريخ سابق لهذه النخلة — تحليل أول مرة.",
            }

        return {
            "palm_id": palm_id,
            "has_history": True,
            "variety": history.get("variety"),
            "location": history.get("location"),
            "last_analysis": history["previous_analysis"][-max_items:],
            "last_recommendations": history["previous_recommendations"][-max_items:],
            "farmer_action": history.get("farmer_action"),
            "next_check_date": history.get("next_check_date"),
        }

    def list_all_palms(self) -> List[str]:
        """إرجاع كل معرفات النخيل المسجلة في الذاكرة."""
        return list(self._load().keys())


if __name__ == "__main__":
    # تجربة سريعة توضح تدفق الاستخدام الكامل من إنشاء النخلة
    # وحتى تسجيل قرار المزارع، بنفس مثال Palm ID: P-014 من التوثيق.
    memory = PalmMemory()

    memory.create_palm("P-014", variety="Sukari", location="Block A - Row 3")

    memory.add_analysis(
        "P-014",
        {
            "fruit_density": "high",
            "growth_stage": "kimri",
            "vision_confidence": 0.86,
            "image_url": "https://example.com/palms/P-014/latest.jpg",
        },
    )

    memory.add_recommendation(
        "P-014",
        {
            "thinning_needed": True,
            "thinning_percentage": 25,
            "confidence_score": 0.82,
            "priority": "high",
            "next_check_days": 14,
        },
    )

    memory.record_farmer_action("P-014", "accepted", note="تم تنفيذ التخفيف بنسبة 25%")

    print(json.dumps(memory.get_recent_context("P-014"), ensure_ascii=False, indent=2))
