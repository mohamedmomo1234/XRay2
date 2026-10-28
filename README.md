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