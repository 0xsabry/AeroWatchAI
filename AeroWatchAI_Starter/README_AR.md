# AeroWatch AI — Starter Kit للمحاكاة البصرية

هذه الحزمة تبدأ بمولّد مشاهد جوية اصطناعية داخل Blender، وتصدر صورًا مع تسميات YOLO لفئات بصرية عامة: `drone`, `bird`, `aircraft`, `balloon`.

> هذه النسخة تثبت خط إنتاج بيانات البحث فقط. الداتا الاصطناعية المحدودة لا تثبت دقة واقعية، ولا ينبغي استخدامها لاتخاذ قرارات سلامة أو تشغيلية. لا يتضمن المشروع تتبع أسلحة أو توجيه/اعتراض.

## 1) ثبّت Blender

استخدم الإصدار LTS الحالي من الصفحة الرسمية:
https://www.blender.org/download/LTS/

## 2) انسخ ملفات الحزمة إلى مجلد المشروع

ضع المجلدات والملفات بجانب `.venv` في `AeroWatchAI`، بحيث يكون المسار `AeroWatchAI/blender_scripts/generate_synthetic_data.py`.

## 3) ولّد مجموعة تجريبية صغيرة

افتح PowerShell في مجلد المشروع، فعّل البيئة، ثم شغّل:

```powershell
.\.venv\Scripts\Activate.ps1
```

عدّل مسار Blender إذا اختلفت نسخة التثبيت. مثال شائع:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python .\blender_scripts\generate_synthetic_data.py -- --samples 40
```

إذا كانت نسخة Blender لديك بمسار مختلف، ابحث عن `blender.exe` وعدّل المسار فقط. يمكن فتح Blender GUI كذلك، لكن أمر الخلفية أسهل لتوليد الصور.

ستجد الصور والتسميات في:

- `data/synthetic_v1/images/train/`
- `data/synthetic_v1/images/val/`
- `data/synthetic_v1/labels/train/`
- `data/synthetic_v1/labels/val/`
- `data/synthetic_v1/dataset.yaml`
- `scenes/aerowatch_synthetic_sky.blend`

ابدأ بـ 40 صورة للتأكد من المسارات وأن الصور واضحة والتسميات تغطي الأجسام. بعد الفحص يمكن رفع العدد مثلًا إلى 300–1000 وتوسيع تنوع المشاهد. الأعداد الصغيرة مجرد تحقق من سير العمل، وليست كافية لإثبات نموذج قوي.

## 4) افحص الصور والتسميات قبل التدريب

افتح بعض الصور من مجلد `images`، وتأكد أن ملف `.txt` الذي يحمل نفس الاسم موجود في `labels` وأن الصندوق المعياري يطابق الجسم. عند اكتشاف مشكلة في الصناديق أو المشهد، أصلح المولّد قبل تدريب النموذج.

## 5) ثبّت متطلبات التدريب عند الحاجة

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

تأكد أن CUDA ما زالت `True` بعد التثبيت. بعدها، وبعد مراجعة تنوع البيانات، يمكن تجربة baseline:

```powershell
python train_yolo.py
```

يستخدم السكربت نموذجًا صغيرًا و`batch=2` لتقليل ضغط الذاكرة على RTX 3050 Laptop GPU 4GB. قد تحتاج خفض `batch` إلى 1 أو `imgsz` إلى 512 إذا نفدت ذاكرة GPU.

## حدود مهمة

- الخلفيات والأجسام المبنية بإجرائية داخل Blender لا تمثل كل تنوع العالم الحقيقي.
- لا تخلط صورًا متجاورة أو شبه متطابقة بين التدريب والاختبار.
- ارفع التنوع الواقعي وراجع التسميات ثم قيّم Precision/Recall وmAP على مجموعة اختبار مستقلة.
- نتائج النموذج على بيانات Blender لا تتنبأ وحدها بأدائه على كاميرات حقيقية.
