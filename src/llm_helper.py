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





#########################
# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("❌ لم يتم العثور على GROQ_API_KEY في ملف .env")

# client = Groq(api_key=api_key)

# # اختار اسم موديل موجود فعلاً في قائمة Groq (console.groq.com/docs/models)
# MODEL_NAME = "اسم-الموديل-من-قائمة-Groq"


# def build_prompt(age, gender, symptoms, model_prediction, confidence, context_chunks):
#     context = "\n\n---\n\n".join(context_chunks) if context_chunks else "لا توجد مصادر متاحة."

#     return f"""
# أنت مساعد طبي يعمل تحت إشراف طبيب متخصص. اكتب تقرير مبدئي مختصر ومنظم.

# بيانات الحالة:
# - العمر: {age} سنة
# - النوع: {gender}
# - الأعراض: {symptoms}
# - نتيجة نموذج الذكاء الاصطناعي على الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)

# المصادر الطبية المرجعية (استخدم هذه المعلومات فقط):
# {context}

# قواعد إلزامية:
# 1. النتيجة مبدئية دائماً وتحتاج مراجعة طبيب. لا تكتب تشخيصاً نهائياً.
# 2. لا تذكر أي معلومة طبية غير موجودة في المصادر. لو مش موجودة اكتب "غير متوفر في المصادر".
# 3. لا تكتب جرعات ولا أسماء أدوية محددة. اكتب العلاج بشكل عام واطلب استشارة الطبيب أو الصيدلي.
# 4. الأشعة وحدها لا تحدد هل الالتهاب بكتيري أو فيروسي. اذكر ذلك صراحة لو الحالة التهاب.
# 5. لو الأعراض خطيرة (ضيق نفس شديد، ألم في الصدر، زرقة، حرارة عالية مستمرة)، اذكر التوجه للطوارئ في أول التقرير.
# 6. لا تذكر أورام أو أي حالة غير الطبيعي والالتهاب الرئوي.

# اكتب التقرير بالعربية بالترتيب ده:
# 1. ملخص الحالة والنتيجة المبدئية.
# 2. الأسباب المحتملة (من المصادر).
# 3. الأعراض المرتبطة وعلامات الخطر.
# 4. الخطة العلاجية العامة (إرشادية).
# 5. نصائح الرعاية المنزلية.
# 6. متى يجب التوجه للطوارئ فوراً.
# """


# def generate_medical_report(age, gender, symptoms, model_prediction, confidence, context_chunks):
#     prompt = build_prompt(age, gender, symptoms, model_prediction, confidence, context_chunks)
#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#             max_tokens=800,
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"عذراً، حدث خطأ أثناء توليد التقرير: {str(e)}"






######################

# import os
# from dotenv import load_dotenv
# from groq import Groq

# # تحميل المتغيرات من ملف .env المحلي
# load_dotenv()

# # استدعاء الـ API Key بطريقة آمنة
# api_key = os.getenv("GROQ_API_KEY")

# if not api_key:
#     raise ValueError("❌ تنبيه: لم يتم العثور على مفتاح GROQ_API_KEY في ملف .env!")

# # تهيئة عميل Groq
# client = Groq(api_key=api_key)

# def generate_medical_report(age, gender, symptoms, model_prediction, confidence):
#     prompt = f"""
#     أنت طبيب استشاري متخصص في الأشعة والأمراض الصدرية. قم إعداد تقرير طبي احترافي بناءً على البيانات التالية:
#     - عمر المريض: {age} سنة
#     - النوع: {gender}
#     - الأعراض التي يشتكي منها: {symptoms}
#     - نتيجة الذكاء الاصطناعي لتحليل الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)
    
#     قم بتضمين الآتي في التقرير (باللغة العربية وبصياغة طبية احترافية):
#     1. تحليل مبدئي للحالة وربط الأعراض بالسن والنتيجة المتوقعة.
#     2. تفسير سبب الإصابة المحتمل (هل هو مرتبط بعدوى، مناعة، أو عوامل عمرية).
#     3. الخطة العلاجية المقترحة المبدئية (للإرشاد العام وليست وصفة نهائية).
#     4. نصائح الراحة المنزلية والمشروبات المفيدة (مثل المشروبات الدافئة، الأطعمة الداعمة للمناعة).
#     5. متى يجب التوجه للطوارئ فوراً؟
#     """
    
#     try:
#         completion = client.chat.completions.create(
#             # استخدام موديل قوي ودقيق للتقارير الطبية
#             model="qwen/qwen3.8-27b",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=600  # تقييد عدد الستوكينز لتوفير الاستهلاك وسرعة الاستجابة
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"عذراً، حدث خطأ أثناء توليد التقرير بالذكاء الاصطناعي: {str(e)}"




###################


# import os
# from groq import Groq

# # ضع مفتاح الـ Groq الخاص بك أو اجعله في Environment Variable
# client = Groq(api_key="ضع_مفتاح_الـ_Groq_هنا")

# def generate_medical_report(age, gender, symptoms, model_prediction, confidence):
#     prompt = f"""
#     أنت طبيب استشاري متخصص في الأشعة والأمراض الصدرية. قم بإعداد تقرير طبي احترافي بناءً على البيانات التالية:
#     - عمر المريض: {age} سنة
#     - النوع: {gender}
#     - الأعراض التي يشتكي منها: {symptoms}
#     - نتيجة الذكاء الاصطناعي لتحليل الأشعة: {model_prediction} (نسبة الثقة: {confidence:.2f}%)
    
#     قم بتضمين الآتي في التقرير (باللغة العربية وبصياغة طبية احترافية):
#     1. تحليل مبدئي للحالة وربط الأعراض بالسن والنتيجة المتوقعة.
#     2. تفسير سبب الإصابة المحتمل (هل هو مرتبط بعدوى، مناعة، أو عوامل عمرية).
#     3. الخطة العلاجية المقترحة المبدئية (للإرشاد العام وليست وصفة نهائية).
#     4. نصائح الراحة المنزلية والمشروبات المفيدة (مثل المشروبات الدافئة، الأطعمة الداعمة للمناعة).
#     5. متى يجب التوجه للطوارئ فوراً؟
#     """
    
#     try:
#         completion = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=1000
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"عذراً، حدث خطأ أثناء توليد التقرير بالذكاء الاصطناعي: {str(e)}"