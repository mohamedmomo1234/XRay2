import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ لم يتم العثور على GROQ_API_KEY في ملف .env")

client = Groq(api_key=api_key)


MODEL_NAME = "qwen/qwen3.8-27b"


def build_prompt(age, gender, symptoms, model_prediction, confidence, context_chunks):
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "لا توجد مصادر متاحة."

    return f"""
أنت مساعد طبي يعمل تحت إشراف طبيب متخصص. اكتب تقرير مبدئي مختصر ومنظم.

بيانات الحالة:
- العمر: {age} سنة
- النوع: {gender}
- الأعراض: {symptoms if symptoms.strip() else "لم يذكر المريض أعراضاً"}
- نتيجة نموذج الذكاء الاصطناعي على الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)

المصادر الطبية المرجعية (استخدم هذه المعلومات فقط):
{context}

قواعد إلزامية:
1. النتيجة مبدئية دائماً وتحتاج مراجعة طبيب. لا تكتب تشخيصاً نهائياً.
2. لا تذكر أي معلومة طبية غير موجودة في المصادر. لو مش موجودة اكتب "غير متوفر في المصادر".
3. لا تكتب جرعات ولا أسماء أدوية محددة. اكتب العلاج بشكل عام واطلب استشارة الطبيب أو الصيدلي.
4. الأشعة وحدها لا تحدد هل الالتهاب بكتيري أو فيروسي. اذكر ذلك صراحة لو الحالة التهاب.
5. لو الأعراض خطيرة (ضيق نفس شديد، ألم في الصدر، زرقة، حرارة عالية مستمرة)، اذكر التوجه للطوارئ في أول التقرير.
6. لا تذكر أورام أو أي حالة غير الطبيعي والالتهاب الرئوي.
7. الجملة الإلزامية الأخيرة: "هذا التقرير مبدئي ويجب مراجعة طبيب مختص قبل اتخاذ أي قرار علاجي."

اكتب التقرير بالعربية بالترتيب ده:
1. ملخص الحالة والنتيجة المبدئية.
2. الأسباب المحتملة (من المصادر).
3. الأعراض المرتبطة وعلامات الخطر.
4. الخطة العلاجية العامة (إرشادية).
5. نصائح الرعاية المنزلية.
6. متى يجب التوجه للطوارئ فوراً.
7. الجملة الإلزامية الأخيرة: "هذا التقرير مبدئي ويجب مراجعة طبيب مختص قبل اتخاذ أي قرار علاجي."
"""


def generate_medical_report(age, gender, symptoms, model_prediction, confidence, context_chunks):
    prompt = build_prompt(age, gender, symptoms, model_prediction, confidence, context_chunks)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"عذراً، حدث خطأ أثناء توليد التقرير: {str(e)}"
