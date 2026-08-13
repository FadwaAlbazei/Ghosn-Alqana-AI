# 🌴 Vision Agent — Autonomous Monitoring (v4, simulated)
This version runs as a **simulated autonomous monitor**: you start it once, and it keeps capturing and analyzing frames on its own on a timer — you don't trigger each capture manually. It only surfaces a clear **alert** when the detected growth stage (i.e. ripeness/color stage) matches the target you set, or when excessive fruit load is detected.

**Important honesty note about Colab:** this is a *simulation* of autonomous monitoring, not a true always-on background service. Colab only runs code while:
- the notebook tab stays open in your browser, and
- the cell is actively running (once it finishes or the runtime disconnects, monitoring stops).

For a real production deployment (camera permanently mounted at the farm, running 24/7 even with the browser closed), this loop would need to run on a dedicated server/device pointed at an IP/RTSP camera — the analysis logic (Steps 3–7 below) stays the same, only the capture step changes.

**Pipeline (within one monitoring session):**
```
Start once → camera opens (one permission prompt) → loop:
   capture frame → quality check → Gemini Vision → check against target
   → if match: 🔔 ALERT  → else: wait → capture again
→ stop when matched or max rounds reached
```

**Setup required:** Add your Gemini API key to Colab Secrets (🔑 icon, left sidebar) under the name `GEMINI_API_KEY` before running.
## 1. Install dependencies
!pip install -q google-generativeai pillow pydantic
## 2. Camera access — open once, capture many frames automatically
The key difference from the previous version: the browser camera permission is requested **only once**. After that, `capture_frame_batch()` keeps grabbing frames on its own at a fixed interval, entirely inside the browser — no repeated manual capture, no repeated permission prompts.
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

            await new Promise((resolve) => setTimeout(resolve, 1200)); // let camera settle

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
## 3. Output schema
Unchanged: the Vision Agent only reports observations. No thinning decision here.
from pydantic import BaseModel
from typing import Literal, Optional

class VisionAnalysis(BaseModel):
    valid_image: bool
    fruit_density: Optional[Literal["low", "medium", "high"]] = None
    fruit_size: Optional[Literal["small", "medium", "large"]] = None
    growth_stage: Optional[str] = None       # kimri / khalal / rutab / tamr
    excessive_load: Optional[bool] = None
    visual_observations: Optional[str] = None   # Arabic free-text description
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
Same as before: observation-only, `visual_observations` in Arabic, enum fields in English for stable downstream logic.
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
## 7. Analyze a single frame and blend confidence
import json

def analyze_palm_image(image_path: str) -> dict:
    if not is_valid_image(image_path):
        return VisionAnalysis(
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
                valid_image=False,
                error=data.get("visual_observations", "الصورة غير صالحة للتحليل")
            ).model_dump()

        model_confidence = float(data.get("model_confidence", 0.5))
        final_confidence = round(0.5 * model_confidence + 0.5 * quality_score, 2)

        result = VisionAnalysis(
            valid_image=True,
            fruit_density=data.get("fruit_density"),
            fruit_size=data.get("fruit_size"),
            growth_stage=data.get("growth_stage"),
            excessive_load=data.get("excessive_load"),
            visual_observations=data.get("visual_observations"),
            model_confidence=model_confidence,
            image_quality_score=quality_score,
            confidence=final_confidence,
        )
        return result.model_dump()

    except json.JSONDecodeError:
        return VisionAnalysis(
            valid_image=False,
            image_quality_score=quality_score,
            error="تعذر تحويل استجابة النموذج إلى JSON"
        ).model_dump()
    except Exception as e:
        return VisionAnalysis(
            valid_image=False,
            image_quality_score=quality_score,
            error=f"خطأ أثناء التحليل: {str(e)}"
        ).model_dump()
## 8. Autonomous monitoring loop — the core of what you asked for
You start this **once**. It then automatically:
1. Captures a batch of frames (single permission prompt from Step 2).
2. Analyzes each frame.
3. Checks each result against your target condition (`target_growth_stage` and/or `alert_on_excessive_load`).
4. The moment a match is found → prints a clear 🔔 alert and stops (you didn't have to trigger anything after the first run).
5. If nothing matches in a round, it automatically starts another round — no manual re-run needed — until `max_rounds` is reached.

Set `target_growth_stage` to whichever stage means "ready" for your use case (e.g. `"rutab"` or `"tamr"`), or leave it `None` to only alert on excessive load.
def run_autonomous_monitoring(
    target_growth_stage: str = "rutab",
    alert_on_excessive_load: bool = True,
    min_confidence: float = 0.5,
    batch_size: int = 3,
    interval_ms: int = 3000,
    max_rounds: int = 5,
):
    """Simulated autonomous monitoring. Runs multiple rounds automatically, each round
    capturing `batch_size` frames without any manual trigger, until a target condition
    is detected or `max_rounds` is reached. Returns (alert_result, all_results_log)."""
    all_results = []

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== الجولة {round_num} من {max_rounds}: جاري التقاط {batch_size} صور تلقائيًا... ===")
        frame_paths = capture_frame_batch(num_frames=batch_size, interval_ms=interval_ms)

        for i, frame_path in enumerate(frame_paths):
            print(f"تحليل الصورة {i + 1} من الجولة {round_num}...")
            result = analyze_palm_image(frame_path)
            result["round"] = round_num
            result["frame_index"] = i
            all_results.append(result)

            if not result.get("valid_image"):
                print(f"  ⚠️ صورة غير صالحة: {result.get('error')}")
                continue

            confidence_ok = (result.get("confidence") or 0) >= min_confidence
            stage_match = target_growth_stage is not None and result.get("growth_stage") == target_growth_stage
            load_match = alert_on_excessive_load and result.get("excessive_load") is True

            if confidence_ok and (stage_match or load_match):
                reason = []
                if stage_match:
                    reason.append(f"مرحلة النضج وصلت للهدف ({target_growth_stage})")
                if load_match:
                    reason.append("تم رصد حمل زائد على العذق")
                print("\n🔔 تنبيه — تم رصد الحالة المطلوبة!")
                print("السبب:", " و ".join(reason))
                print("تفاصيل الرصد:", result.get("visual_observations"))
                print(f"درجة الثقة: {result.get('confidence')}")
                return result, all_results

        print(f"لا توجد مطابقة بعد بهذي الجولة، جاري الانتقال للجولة التالية تلقائيًا...")

    print("\n⏹️ انتهت جولات المراقبة (بلغنا الحد الأقصى) بدون رصد الحالة المطلوبة.")
    return None, all_results
## 9. Start monitoring
Run this **one cell** and the system takes it from there automatically — capturing, analyzing, and checking against your target on its own, round after round, until it either alerts you or reaches `max_rounds`.
alert_result, monitoring_log = run_autonomous_monitoring(
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
- **This is a simulation, not a real background service.** It only runs while this Colab tab is open and this cell is executing. For production (camera mounted permanently at the farm, running unattended), the capture step needs to move to a dedicated always-on machine reading from an IP/RTSP camera — the rest of the logic (schema, prompt, confidence blending, alert condition) carries over unchanged.
- **The alert condition is configurable**: `target_growth_stage` (maps to the ripeness/color stage you're watching for) and `alert_on_excessive_load`. Add more conditions the same way if needed (e.g. `fruit_density == "high"`).
- **`confidence` is blended**, not invented: `0.5 * model_confidence + 0.5 * image_quality_score`. Tune the weights after testing on real images.
- **This agent never recommends thinning** — only reports observations and raises an alert when they match your target. The Decision Agent downstream should combine this with sensor data (e.g. color sensor) before any final action, and fall back to human review when confidence is low.
- `monitoring_log` holds every analyzed frame across all rounds — useful for debugging or showing the full monitoring history in the demo.
