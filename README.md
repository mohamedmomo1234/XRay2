# MediAI: نظام مساعد لتحليل أشعة الصدر (CDSS - Prototype)

نموذج أولي لمشروع تخرج جامعي، يحلل صورة أشعة الصدر ويصنفها **طبيعية** أو **التهاب رئوي**، ثم يعرض خريطة Grad-CAM وتقريراً مبدئياً مبنياً على مصادر طبية محددة.

> **تنبيه:** هذا النظام ليس أداة تشخيص طبي، ولا يصلح للاستخدام السريري. النتائج مبدئية وتحتاج مراجعة طبيب متخصص.

---

## 1. حدود النظام

- يكتشف **الالتهاب الرئوي فقط** مقابل الأشعة الطبيعية.
- لا يكتشف الأورام ولا أمراض الصدر الأخرى.
- لا يفرّق بين الالتهاب **البكتيري** والفيروسي، رغم أن الداتاسيت فيها الاتنين.
- يعتمد على داتاسيت واحدة، وأداؤه على صور من مصادر أخرى غير معروف.

---

## 2. الداتاسيت

### المصدر

- **الاسم:** Chest X-Ray: bacterial / viral pneumonia, normal
- **الرابط:** [kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal](https://www.kaggle.com/datasets/kostasdiamantaras/chest-xrays-bacterial-viral-pneumonia-normal)
- **المصدر الأصلي:** Chest X-Ray Images (Pneumonia) dataset، المنشورة في مجلة Cell (2018)
- **الترخيص:** CC BY 4.0

### الفئات (Labels)

| class_id | المعنى | عدد الصور في التدريب |
|---|---|---|
| 0 | Normal (لا يوجد مرض) | 1227 |
| 1 | Bacterial pneumonia (التهاب بكتيري) | 2238 |
| 2 | Viral pneumonia (التهاب فيروسي) | 1207 |

### التحويل للتصنيف الثنائي

في المشروع بنحول الفئات لتصنيف ثنائي:

- `0` → **NORMAL** (y = 0)
- `1` أو `2` → **PNEUMONIA** (y = 1)

النتيجة: 1227 طبيعي مقابل 3445 التهاب، وده عدم توازن واضح، فبنستخدم `pos_weight` في دالة الخسارة.

### التقسيم

الداتاسيت فيها مجلدين:

- `train_images/`: 4672 صورة، ليها labels في `labels_train.csv`.
- `test_images/`: 1168 صورة، **مفيهاش labels**، فمش بنستخدمها في التقييم.

عشان كده قسّمنا `labels_train.csv` بنفسنا بطريقة **stratified** (تحافظ على نسبة الفئات في كل جزء):

| الجزء | النسبة | عدد تقريبي |
|---|---|---|
| Train | 70% | ~3270 |
| Validation | 15% | ~701 |
| Test | 15% | 701 (NORMAL: 184، PNEUMONIA: 517) |

الـ Validation بيُستخدم لاختيار أحسن epoch وحساب الـ threshold. الـ Test بيتقيّم مرة واحدة في الآخر.

---

## 3. استخراج البيانات (Data Extraction)

الملف المضغوط اتنزل من Kaggle ورُفع على Google Colab، وبعدين اتفك ضغطه بالأمر ده:

```bash
unzip -q chest-xrays-bacterial-viral-pneumonia-normal.zip -d /content/data

/content/data/
├── labels_train.csv          (أعمدة: file_name, class_id)
├── train_images/
│   └── train_images/         (4672 صورة)
└── test_images/
    └── test_images/          (1168 صورة)




Backbone: EfficientNet-B0 مدرب مسبقاً على ImageNet (EfficientNet_B0_Weights.DEFAULT).
Head الجديد:
Dropout(p=0.3)
Linear(1280 → 1)   ← logit واحد
Output: logit، والـ sigmoid بيتطبق وقت التقييم، مش جوه الموديل.
دالة الخسارة
BCEWithLogitsLoss مع pos_weight = n_normal / n_pneumonia (محسوب من الـ train فقط) عشان نعوّض عدم التوازن.








Optimizer: AdamW، weight_decay = 1e-4.
Scheduler: CosineAnnealingLR.
Batch size: 32.
Seed: 42 (للتكرار).
الـ Data Augmentation (على الـ train بس)
RandomHorizontalFlip
RandomRotation (±10°)
RandomAffine (translate 5%، scale 0.95 لـ 1.05)
ColorJitter (brightness 0.2، contrast 0.2)
الـ validation والـ test بياخدوا تحويل بسيط بدون augmentation.
5. ضبط الموديل (Tuning)
الـ Checkpoint
بنحفظ أحسن موديل حسب val AUC عبر كل الـ epochs، مش آخر epoch.
الـ Threshold
الـ threshold الافتراضي 0.5 مش مناسب مع عدم التوازن، فبنختاره من الـ validation عن طريق أعلى F1 على منحنى Precision-Recall. القيمة بتتحفظ في models/threshold.json وبيقرأها التطبيق.





النتائج
الـ baseline (قبل الإصلاح): NORMAL recall = 0.20، ROC-AUC = 0.8456.
النتيجة الحالية على الـ Test (701 صورة):
الفئة
Precision
Recall
F1
Support
NORMAL
0.97
0.91
0.94
184
PNEUMONIA
0.97
0.99
0.98
517
Accuracy


0.97
701
Confusion Matrix: [[167, 17], [5, 512]]
ROC-AUC: 0.9954
ملاحظات أمانة على النتائج:
الأرقام دي على test من نفس الداتاسيت، مش على حالات من مستشفى.
الـ split مش مبني على المريض (patient-level)، والداتاسيت مش بتعرض معرّف المريض، فممكن يكون فيه صور لنفس المريض في الـ train والـ test، وده يرفع الأرقام.
لم يتم عمل hyperparameter search منهجي. القيم اختيرت بناءً على الممارسات الشائعة مع EfficientNet.





project/
├── app.py                    # واجهة Streamlit (لوحة المريض ولوحة الطبيب)
├── models/
│   ├── efficientnet_pneumonia.pth   # موديل التشخيص
│   ├── threshold.json               # الـ threshold المختار من الـ validation
│   └── xray_validator.pth           # موديل التحقق من الصورة (قيد التطوير)
├── knowledge/
│   ├── normal.md             # معلومات عن النتيجة الطبيعية
│   ├── pneumonia.md          # معلومات عن الالتهاب الرئوي
│   └── general.md            # نصائح عامة وعلامات الطوارئ
├── src/
│   ├── model.py              # تحميل الموديل، transforms، Grad-CAM
│   ├── validator.py          # مدقق نوع الصورة
│   ├── rag.py                # استرجاع المعلومات بـ TF-IDF
│   └── llm_helper.py         # توليد التقرير عبر Groq
└── .env                      # GROQ_API_KEY




المعلومات الطبية (RAG)
الاسترجاع بـ TF-IDF، ويختار المقاطع حسب النتيجة (NORMAL أو PNEUMONIA) وأعراض المريض.
المحتوى مكتوب يدوياً من مصادر عامة (WHO، CDC، MedlinePlus)، ويحتاج مراجعة طبية قبل أي استخدام.
توليد التقرير
LLM عبر Groq API.
البرومبت بيمنع التشخيص النهائي، وبيمنع ذكر أدوية أو جرعات، وبيمنع تحديد نوع الالتهاب.
كل تقرير بينتهي بجملة إلزامية: "الأشعة لا تكشف أمراض أخرى غير الالتهاب الرئوي، ويحتاج الفحص الكامل مراجعة طبيب."
8. التشغيل
المتطلبات
pip install torch torchvision streamlit pillow opencv-python scikit-learn pandas groq python-dotenv
# اختياري لقراءة PDF
pip install pypdf
إعداد المفتاح
أنشئ ملف .env في جذر المشروع:
GROQ_API_KEY=your_key_here
التشغيل
cd مسار-المشروع
streamlit run app.py
لازم الأمر يتشغل من جذر المشروع، لأن المسارات نسبية (models/، knowledge/).



**demo by Mohamed said abdalaziz










#############################################
# 🫁 Chest X-Ray AI Diagnostic & Clinical Decision Support Platform


> نظام ذكاء اصطناعي هجين (Hybrid AI System) يدمج بين الرؤية الحاسوبية (Computer Vision) والنماذج اللغوية الكبيرة (LLMs) لتحليل أشعة الصدر السينية وتوليد تقارير طبية آلية.

---

## 🏗️ System Architecture

يعتمد النظام على بنية هجينة متكاملة لضمان دقة التشخيص وسرعة توليد التقارير:
1. **Vision Model :** نموذج `EfficientNet`  مُدرب خصيصاً لتصنيف صور الأشعة السينية (`Normal` مقابل `Pneumonia`) مع تطبيق تقنية **Grad-CAM** لتوليد خريطة حرارية توضح مكان الإصابة بدقة.
2. **Generative LLM (موتور اللغة والتقارير):** استخدام نماذج **Llama (عبر Groq API)** لتحويل المخرجات العددية والتشخيصية إلى تقارير طبية احترافية تتضمن التحليل المبدئي، الأسباب المحتملة، خطة العلاج الإرشادية، ونصائح الراحة.
3. **Uncertainty Decision Logic:** نظام عتبات ذكي (Confidence Thresholding) للتعامل مع الحالات الغامضة وتقليل الإنذارات الكاذبة (False Positives).

---

## 🌟 Key Features

* **لوحات تحكم أدوار المستخدمين (Role-based Dashboards):** واجهات مخصصة لكل من المرضى والأطباء.
* **التشخيص الفوري والصور التوضيحية:** عرض نتيجة الفحص مع نسبة الثقة وخريطة Grad-CAM الحرارية.
* **توليد تقارير طبية تلقائية:** صياغة تقارير طبية باللغة العربية بأسلوب استشاري متخصص عبر الـ LLM.
* **الأمان والخصوصية:** إدارة المفاتيح السرية عبر ملفات البيئة (`.env`) وحماية بيانات التشغيل.

---

## 🧰 Tech Stack

* **Programming Language:** Python 3.11
* **Deep Learning & Computer Vision:** PyTorch, torchvision, EfficientNet, OpenCV, Grad-CAM
* **GenAI & LLM:** Groq API (qwen/qwen3.8-27b)
* **Frontend & UI:** Streamlit
* **Environment & Security:** python-dotenv, Git / GitHub

---

## 📁 Project Structure 

```text
XRayChest/
│
├── models/
│   └── efficientnet_pneumonia.pth
│
├── src/
│   ├── model.py        
│   └── llm_helper.py               
│
├── .env                       
├── .gitignore                      
├── app.py                          
└── requirements.txt               

**by Mohamed Said abdalaziz**
