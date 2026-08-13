# 🌴 Vision Agent — Autonomous Monitoring (v5)
Simulated autonomous monitor: start it once, and it keeps capturing and analyzing frames on its own on a timer, tagging every result with a `palm_id` so it can be matched to the right tree's record in Memory/DB. It only surfaces a clear alert when the detected growth stage matches your target, or when excessive fruit load is detected.

**Matches the architecture:**
```
[Camera Image] + [Environmental Sensors]
        ↓
1) VISION AGENT — image analysis → visual_features (JSON)
        ↓
2) AGRICULTURE AGENT — RAG + agronomic rules → diagnosis
        ↓
3) DECISION AGENT — thinning % + confidence + reasoning
        ↓
[DASHBOARD] ↔ [PALM MEMORY / DATABASE]
```
This notebook implements step 1 only. Its output is a `visual_features` JSON tagged with `palm_id`, ready to hand to the Agriculture Agent.

**Important honesty note about Colab:** this is a *simulation* of autonomous monitoring, not a true always-on background service — it only runs while this tab stays open and the cell is executing. A real deployment would move the capture step (Section 2) to a dedicated server/device connected to a fixed IP/RTSP camera per palm; everything else stays the same.

**Output language:** code/comments in English; `visual_observations` and `error` fields in Arabic (shown in the app); enum-style fields stay in English for stable downstream logic.

**Setup required:** Add your Gemini API key to Colab Secrets (🔑 icon, left sidebar) under the name `GEMINI_API_KEY`.
## 1. Install dependencies
!pip install -q google-generativeai pillow pydantic
## 2. Camera access — open once, capture many frames automatically
Browser camera permission is requested **once**. `capture_frame_batch()` then keeps grabbing frames automatically at a fixed interval, entirely inside the browser.
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import time

def capture_frame_batch(num_frames: int = 3, interval_ms: int = 3000, quality: float = 0.9, prefix: str = "frame"):
    """Opens the browser camera ONCE, then automatically captures `num_frames` photos
    spaced `interval_ms` apart, all within a single permission grant. Returns the list
    of saved file paths."""
    js = Javascript('''
        async function capturePhotoSeries(numFrames, intervalMs, quality) {
            const div = document.createElement('div');
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            await new Promise((resolve) => setTimeout(resolve, 1200));

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            const frames = [];
            for (let i = 0; i < numFrames; i++) {
                ctx.drawImage(video, 0, 0);
                frames.push(canvas.toDataURL('image/jpeg', quality));
                if (i < numFrames - 1) {
                    await new Promise((resolve) => setTimeout(resolve, intervalMs));
                }
            }

            stream.getVideoTracks()[0].stop();
            div.remove();
            return frames;
        }
        capturePhotoSeries(%d, %d, %f)
    ''' % (num_frames, interval_ms, quality))

    display(js)
    data_urls = eval_js('capturePhotoSeries({}, {}, {})'.format(num_frames, interval_ms, quality))

    paths = []
    for i, data_url in enumerate(data_urls):
        binary = b64decode(data_url.split(',')[1])
        path = f"{prefix}_{int(time.time())}_{i}.jpg"
        with open(path, 'wb') as f:
            f.write(binary)
        paths.append(path)
    return paths
## 3. Output schema — matches the "visual_features" contract
`palm_id` identifies which tree this observation belongs to, so the Agriculture/Decision Agent and Memory/DB can attach it to the right record. `visual_features` is a nested object matching your architecture table exactly, now including `spikelet_count` (تقدير عدد الشماريخ), which was missing before.
from pydantic import BaseModel
from typing import Literal, Optional

class VisualFeatures(BaseModel):
    fruit_density: Optional[Literal["low", "medium", "high"]] = None
    spikelet_count: Optional[int] = None        # تقدير عدد الشماريخ
    fruit_size: Optional[Literal["small", "medium", "large"]] = None
    growth_stage: Optional[str] = None          # kimri / khalal / rutab / tamr
    excessive_load: Optional[bool] = None
    visual_observations: Optional[str] = None   # Arabic free-text description


class VisionAnalysis(BaseModel):
    palm_id: Optional[str] = None
    valid_image: bool
    visual_features: Optional[VisualFeatures] = None
    model_confidence: Optional[float] = None
    image_quality_score: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None                 # Arabic error message when present
## 4. Image quality validation and scoring
from PIL import Image, ImageFilter
import numpy as np

def is_valid_image(image_path: str, min_size: int = 200) -> bool:
    try:
        img = Image.open(image_path)
        width, height = img.size
        if width < min_size or height < min_size:
            return False
        extrema = img.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            return False
        return True
    except Exception:
        return False


def compute_image_quality_score(image_path: str) -> float:
    """Rough, explainable quality score from 0 to 1 (sharpness + brightness)."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    edges = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = edges.var()
    sharpness_score = min(sharpness / 500.0, 1.0)

    mean_brightness = arr.mean()
    brightness_score = 1.0 - abs(mean_brightness - 128) / 128

    quality_score = round(0.6 * sharpness_score + 0.4 * brightness_score, 2)
    return max(0.0, min(quality_score, 1.0))
## 5. Configure the Gemini API
Add your key to Colab Secrets: 🔑 icon (left sidebar) → Add secret → name `GEMINI_API_KEY`.
import google.generativeai as genai
from google.colab import userdata

GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
## 6. Analysis prompt
Now also asks for `spikelet_count` — the model's best estimate of the number of fruit-bearing strands visible on the bunch.
PROMPT = """
You are a computer vision component that ONLY reports observations about a date palm
fruit bunch image. You do NOT make any harvesting, thinning, or agricultural decisions —
that is handled by a separate downstream agent. Do not include any recommendation,
suggestion, or judgment about what action should be taken.

Analyze the attached image and return the result as JSON ONLY, with no extra text
and no ``` markers, using exactly these fields:

{
  "valid_image": true or false,
  "fruit_density": "low" or "medium" or "high",
  "spikelet_count": your best integer estimate of the number of visible fruit-bearing
    strands (spikelets) on the bunch, or null if not countable in this image,
  "fruit_size": "small" or "medium" or "large",
  "growth_stage": "kimri" or "khalal" or "rutab" or "tamr",
  "excessive_load": true or false,
  "visual_observations": "a short, purely descriptive summary of what you observe, WRITTEN IN ARABIC",
  "model_confidence": a decimal number between 0 and 1 reflecting how clearly you
    could identify the above features in this specific image
}

If the image is unclear, too dark, or does not show a clear date palm fruit bunch,
return valid_image: false, set the other fields to null, and explain why in
visual_observations (in Arabic).
"""
## 7. Analyze a single frame, tag with palm_id, blend confidence
import json

def analyze_palm_image(image_path: str, palm_id: str) -> dict:
    if not is_valid_image(image_path):
        return VisionAnalysis(
            palm_id=palm_id,
            valid_image=False,
            error="جودة الصورة غير كافية (دقة منخفضة أو صورة فارغة/سوداء)"
        ).model_dump()

    quality_score = compute_image_quality_score(image_path)

    try:
        uploaded_file = genai.upload_file(image_path)
        response = model.generate_content([PROMPT, uploaded_file])

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        if not data.get("valid_image", False):
            return VisionAnalysis(
                palm_id=palm_id,
                valid_image=False,
                error=data.get("visual_observations", "الصورة غير صالحة للتحليل")
            ).model_dump()

        model_confidence = float(data.get("model_confidence", 0.5))
        final_confidence = round(0.5 * model_confidence + 0.5 * quality_score, 2)

        features = VisualFeatures(
            fruit_density=data.get("fruit_density"),
            spikelet_count=data.get("spikelet_count"),
            fruit_size=data.get("fruit_size"),
            growth_stage=data.get("growth_stage"),
            excessive_load=data.get("excessive_load"),
            visual_observations=data.get("visual_observations"),
        )

        result = VisionAnalysis(
            palm_id=palm_id,
            valid_image=True,
            visual_features=features,
            model_confidence=model_confidence,
            image_quality_score=quality_score,
            confidence=final_confidence,
        )
        return result.model_dump()

    except json.JSONDecodeError:
        return VisionAnalysis(
            palm_id=palm_id,
            valid_image=False,
            image_quality_score=quality_score,
            error="تعذر تحويل استجابة النموذج إلى JSON"
        ).model_dump()
    except Exception as e:
        return VisionAnalysis(
            palm_id=palm_id,
            valid_image=False,
            image_quality_score=quality_score,
            error=f"خطأ أثناء التحليل: {str(e)}"
        ).model_dump()
## 8. Autonomous monitoring loop
Same behavior as before, now tagging every observation with `palm_id`. Start this once — it captures, analyzes, and checks against your target automatically, round after round, until it alerts you or reaches `max_rounds`.
def run_autonomous_monitoring(
    palm_id: str,
    target_growth_stage: str = "rutab",
    alert_on_excessive_load: bool = True,
    min_confidence: float = 0.5,
    batch_size: int = 3,
    interval_ms: int = 3000,
    max_rounds: int = 5,
):
    """Simulated autonomous monitoring for one palm (identified by palm_id). Runs multiple
    rounds automatically, each capturing `batch_size` frames without manual triggering,
    until a target condition is detected or `max_rounds` is reached.
    Returns (alert_result, all_results_log)."""
    all_results = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== [{palm_id}] الجولة {round_num} من {max_rounds}: جاري التقاط {batch_size} صور تلقائيًا... ===")
        frame_paths = capture_frame_batch(num_frames=batch_size, interval_ms=interval_ms)

        for i, frame_path in enumerate(frame_paths):
            print(f"تحليل الصورة {i + 1} من الجولة {round_num}...")
            result = analyze_palm_image(frame_path, palm_id=palm_id)
            result["round"] = round_num
            result["frame_index"] = i
            all_results.append(result)

            if not result.get("valid_image"):
                print(f"  ⚠️ صورة غير صالحة: {result.get('error')}")
                continue

            features = result.get("visual_features") or {}
            confidence_ok = (result.get("confidence") or 0) >= min_confidence
            stage_match = target_growth_stage is not None and features.get("growth_stage") == target_growth_stage
            load_match = alert_on_excessive_load and features.get("excessive_load") is True

            if confidence_ok and (stage_match or load_match):
                reason = []
                if stage_match:
                    reason.append(f"مرحلة النضج وصلت للهدف ({target_growth_stage})")
                if load_match:
                    reason.append("تم رصد حمل زائد على العذق")
                print(f"\n🔔 تنبيه — النخلة {palm_id}: تم رصد الحالة المطلوبة!")
                print("السبب:", " و ".join(reason))
                print("تفاصيل الرصد:", features.get("visual_observations"))
                print(f"درجة الثقة: {result.get('confidence')}")
                return result, all_results

        print("لا توجد مطابقة بعد بهذي الجولة، جاري الانتقال للجولة التالية تلقائيًا...")

    print(f"\n⏹️ انتهت جولات المراقبة للنخلة {palm_id} (بلغنا الحد الأقصى) بدون رصد الحالة المطلوبة.")
    return None, all_results
## 9. Start monitoring
Set `palm_id` to the tree you're monitoring (this is what lets the Agriculture Agent and Memory/DB attach the result to the right record), and `target_growth_stage` to whatever stage means "ready" for you.
alert_result, monitoring_log = run_autonomous_monitoring(
    palm_id="P-014",               # identifies which palm this observation belongs to
    target_growth_stage="rutab",   # change to the stage that means "ready" for you
    alert_on_excessive_load=True,
    min_confidence=0.5,
    batch_size=3,
    interval_ms=3000,
    max_rounds=5,
)

print("\n--- السجل الكامل لكل الصور المحللة ---")
print(json.dumps(monitoring_log, indent=2, ensure_ascii=False))
---
### Notes for the team
- **Output shape now matches your architecture table**: top-level `palm_id` + `visual_features` (containing `fruit_density`, `spikelet_count`, `fruit_size`, `growth_stage`, `excessive_load`, `visual_observations`) + `confidence` — this is what should be handed directly to the Agriculture Agent.
- **`spikelet_count`** is the model's visual estimate of fruit-bearing strands; it can be `null` when not clearly countable in the frame (e.g. bunch partially occluded). Treat it as an estimate, not a precise count — worth validating against a few real photos before trusting it for agronomic decisions.
- **This is a simulation**, not a real background service — it only runs while this tab is open and the cell is executing. Production would move Section 2's capture step to a dedicated server/device with a fixed IP/RTSP camera per palm.
- **Confidence is blended**, not invented: `0.5 * model_confidence + 0.5 * image_quality_score`. Tune the weights after testing on real images.
- **This agent never recommends thinning** — only reports `visual_features` and raises an alert when they match your target. The Agriculture Agent (RAG + agronomic rules) and Decision Agent handle diagnosis and the thinning percentage/confidence/reasoning, per your architecture.
- `monitoring_log` holds every analyzed frame across all rounds, each tagged with `palm_id` — useful for the Memory/DB history and for debugging.
