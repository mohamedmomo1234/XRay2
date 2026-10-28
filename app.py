import streamlit as st
from PIL import Image
import torch
from src.model import (load_model, get_transforms, GradCAM,
                       generate_heatmap_image, load_threshold)
from src.llm_helper import generate_medical_report
from src.rag import retrieve

# config page
st.set_page_config(page_title="MediAI - Chest Diagnostics Platform", page_icon="🏥", layout="wide")


MODEL_PATH = "models/efficientnet_pneumonia.pth"
THRESHOLD = load_threshold("models/threshold.json")


SCOPE_WARNING = ("⚠️ **حدود النظام:** هذا النظام يكشف **الالتهاب الرئوي فقط** "
                 "(مقابل الأشعة الطبيعية). لا يكشف الأورام ولا أي أمراض أخرى، "
                 "ولا يحدد نوع الالتهاب (بكتيري أو فيروسي).")


@st.cache_resource
def get_cached_model():
    return load_model(MODEL_PATH)


model = get_cached_model()


st.sidebar.title("🏥 نظام تشخيص الصدر الذكي")
role = st.sidebar.selectbox("اختر لوحة التحكم (Role):",
                            ["المريض (Patient Dashboard)", "الطبيب (Doctor Dashboard)"])

st.sidebar.markdown("---")
st.sidebar.info("💡   نظام مساعد ذكي لتحليل أشعة الصدر وتقديم التقارير الطبية.")
st.sidebar.warning("يكتشف الالتهاب الرئوى فقط ولا يكتشف الاورام او امراض اخرى")


# 1.dashboard patient
if role == "المريض (Patient Dashboard)":
    st.title("👤 لوحة تحكم المريض - فحص الأشعة والأعراض")
    st.write("أدخل بياناتك الصحية وارفع صورة الأشعة للحصول على تحليل مبدئي وتقرير شامل.")
    st.info(SCOPE_WARNING)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 البيانات الشخصية والأعراض")
        age = st.number_input("السن (العمر):", min_value=1, max_value=120, value=25)
        gender = st.selectbox("النوع:", ["ذكر", "أنثى"])
        symptoms = st.text_area("ما هي الأعراض التي تعاني منها؟ (مثلاً: كحة جافة، ضيق تنفس، ارتفاع حرارة...)",
                                placeholder="اكتب الأعراض بالتفصيل...")

        uploaded_file = st.file_uploader("اختر صورة أشعة الصدر (JPG/PNG)...",
                                         type=["jpg", "png", "jpeg"])

    with col2:
        st.subheader("🔍 معاينة الأشعة والنتيجة")
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file).convert('RGB')
            st.image(original_image, caption="صورة الأشعة المرفوعة", use_container_width=True)

            if st.button("🚀 ابدأ التحليل الشامل وتوليد التقرير"):
                with st.spinner("جاري معالجة الصورة وتحليل الحالة..."):

                    # 1) prediction model
                    transform = get_transforms()
                    input_tensor = transform(original_image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.sigmoid(output).item()

                    if prob >= THRESHOLD:
                        label_key = "PNEUMONIA"
                        result_label = "PNEUMONIA (التهاب رئوي)"
                        conf_score = prob * 100
                    else:
                        label_key = "NORMAL"
                        result_label = "NORMAL (سليم)"
                        conf_score = (1 - prob) * 100

                    # 2) Grad-CAM & hooks  
                    target_layer = model.features[-1]
                    grad_cam = GradCAM(model, target_layer)
                    try:
                        cam = grad_cam(input_tensor)
                    finally:
                        grad_cam.remove()
                    heatmap_img = generate_heatmap_image(original_image, cam)

                    st.image(heatmap_img, caption="Grad-CAM Heatmap (المناطق اللي أثرت على القرار)",
                             use_container_width=True)

                    # 3) retriver info medical
                    query = f"{label_key} {symptoms}"
                    context = retrieve(label_key, query)

                    # 4) generate report
                    report = generate_medical_report(
                        age, gender, symptoms, result_label, conf_score, context
                    )

                    # 5) save result
                    st.session_state['last_result'] = result_label
                    st.session_state['last_conf'] = conf_score
                    st.session_state['last_report'] = report

    # present medical report
    if 'last_report' in st.session_state:
        st.markdown("---")
        st.subheader("📄 التقرير الطبي المبدئي:")
        st.markdown(st.session_state['last_report'])


# 2. لوحة تحكم الطبيب
elif role == "الطبيب (Doctor Dashboard)":
    st.title("🩺 لوحة تحكم الطبيب - مراجعة الحالات واعتماد التقارير")
    st.write("مراجعة الحالات الواردة من المرضى، وفحص خرائط التمركز والتقارير الذكية.")

    tab1, tab2 = st.tabs(["📁 الحالة الحالية", "📊 إحصائيات النظام"])

    with tab1:
        if 'last_result' in st.session_state:
            st.success("تم استلام أحدث حالة مرفوعة من بوابة المرضى.")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.metric("التشخيص الآلي المبدئي", st.session_state['last_result'])
                st.metric("نسبة الثقة التقنية", f"{st.session_state['last_conf']:.2f}%")
            with col_d2:
                st.warning("⚠️ يرجى مراجعة خريطة Grad-CAM والتأكد من توافقها السريري قبل الاعتماد النهائي.")

            st.subheader("📝 التقرير الطبي المقترح:")
            st.text_area("تعديل أو اعتماد التقرير:",
                         value=st.session_state['last_report'], height=250)

            if st.button("✅ اعتماد التقرير وإرساله لملف المريض"):
                st.success("تم اعتماد التقرير الطبي بنجاح.")
        else:
            st.info("لا توجد حالات جديدة معلقة حالياً.")

    
    with tab2:
       st.subheader("📈 إحصائيات النظام")
       if 'case_log' not in st.session_state:
          st.session_state['case_log'] = []

       log = st.session_state['case_log']
       st.write(f"عدد الحالات المفحوصة في هذه الجلسة: {len(log)}")
       if log:
          normal_n = sum(1 for c in log if c['label'] == 'NORMAL')
          pneu_n = len(log) - normal_n
          st.write(f"طبيعي: {normal_n} | التهاب رئوي: {pneu_n}")
          st.write(f"متوسط الثقة: {sum(c['conf'] for c in log) / len(log):.2f}%")
       else:
          st.info("لا توجد حالات مفحوصة في هذه الجلسة بعد.")

       st.caption("نتائج الاختبار على بيانات التدريب (Test set): Accuracy = 0.97، ROC-AUC = 0.9954، "
               "Recall للالتهاب = 0.99، Recall للطبيعي = 0.91.")



        # st.subheader("📈 إحصائيات النظام ")
        # st.write("عدد الحالات المفحوصة اليوم: 12 حالة")
        # st.write("نسبة دقة التوافق مع التشخيص البشري: 94.2%")

# warning academy
st.markdown("---")
st.caption("⚠️ **تنبيه إخلاء مسؤولية :** هذا النظام نموذج يكتشف الالتهاب الرئوى فقط    "
           "وليس بديلاً عن التشخيص الطبي البشري المتخصص.")




##################
# import streamlit as st
# from PIL import Image
# import torch
# from src.model import (load_model, get_transforms, GradCAM,
#                        generate_heatmap_image, load_threshold)
# from src.llm_helper import generate_medical_report

# # إعدادات الصفحة
# st.set_page_config(page_title="MediAI - Chest Diagnostics Platform", page_icon="🏥", layout="wide")

# # مسار الموديل والـ threshold (الـ threshold اتحسب من الـ validation أثناء التدريب)
# MODEL_PATH = "models/efficientnet_pneumonia.pth"
# THRESHOLD = load_threshold("models/threshold.json")

# @st.cache_resource
# def get_cached_model():
#     return load_model(MODEL_PATH)

# model = get_cached_model()

# # شريط جانبى لتحديد الدور (Dashboard Selector)
# st.sidebar.title("🏥 نظام تشخيص الصدر الذكي")
# role = st.sidebar.selectbox("اختر لوحة التحكم (Role):", ["المريض (Patient Dashboard)", "الطبيب (Doctor Dashboard)"])

# st.sidebar.markdown("---")
# st.sidebar.info("💡 مشروع تخرج - نظام مساعد ذكي لتحليل أشعة الصدر وتقديم التقارير الطبية.")

# # ==========================================
# # 1. لوحة تحكم المريض (Patient Dashboard)
# # ==========================================
# if role == "المريض (Patient Dashboard)":
#     st.title("👤 لوحة تحكم المريض - فحص الأشعة والأعراض")
#     st.write("أدخل بياناتك الصحية واقترح صورة الأشعة للحصول على تحليل مبدئي وتقرير شامل.")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("📋 البيانات الشخصية والأعراض")
#         age = st.number_input("السن (العمر):", min_value=1, max_value=120, value=25)
#         gender = st.selectbox("النوع:", ["ذكر", "أنثى"])
#         symptoms = st.text_area("ما هي الأعراض التي تعاني منها؟ (مثلاً: كحة جافة، ضيق تنفس، ارتفاع حرارة...)",
#                                 placeholder="اكتب الأعراض بالتفصيل...")

#         uploaded_file = st.file_uploader("اختر صورة أشعة الصدر (Chest X-Ray - JPG/PNG)...",
#                                          type=["jpg", "png", "jpeg"])

#     with col2:
#         st.subheader("🔍 معاينة الأشعة والنتيجة")
#         if uploaded_file is not None:
#             original_image = Image.open(uploaded_file).convert('RGB')
#             st.image(original_image, caption="صورة الأشعة المرفوعة", use_container_width=True)

#             if st.button("🚀 ابدأ التحليل الشامل وتوليد التقرير"):
#                 with st.spinner("جاري معالجة الصورة وتحليل الحالة بالذكاء الاصطناعي..."):

#                     # معالجة الصورة والتنبؤ بالموديل
#                     transform = get_transforms()
#                     input_tensor = transform(original_image).unsqueeze(0)

#                     with torch.no_grad():
#                         output = model(input_tensor)
#                         prob = torch.sigmoid(output).item()

#                     if prob >= THRESHOLD:
#                         result_label = "PNEUMONIA (التهاب رئوي)"
#                         conf_score = prob * 100
#                     else:
#                         result_label = "NORMAL (سليم)"
#                         conf_score = (1 - prob) * 100

#                     # توليد Grad-CAM مع تنظيف الـ hooks بعد الاستخدام
#                     target_layer = model.features[-1]
#                     grad_cam = GradCAM(model, target_layer)
#                     try:
#                         cam = grad_cam(input_tensor)
#                     finally:
#                         grad_cam.remove()
#                     heatmap_img = generate_heatmap_image(original_image, cam)

#                     st.image(heatmap_img, caption="Grad-CAM Heatmap (موضع التركيز للالتهاب)",
#                              use_container_width=True)

#                     # حفظ النتائج في Session State لعرضها
#                     st.session_state['last_result'] = result_label
#                     st.session_state['last_conf'] = conf_score
#                     st.session_state['last_report'] = generate_medical_report(
#                         age, gender, symptoms, result_label, conf_score)

#     # عرض التقرير الطبي الناتج
#     if 'last_report' in st.session_state:
#         st.markdown("---")
#         st.subheader("📄 التقرير الطبي المبدئي وتوصيات العلاج والراحة:")
#         st.markdown(st.session_state['last_report'])

# # ==========================================
# # 2. لوحة تحكم الطبيب (Doctor Dashboard)
# # ==========================================
# elif role == "الطبيب (Doctor Dashboard)":
#     st.title("🩺 لوحة تحكم الطبيب - مراجعة الحالات واعتماد التقارير")
#     st.write("هنا يتم مراجعة الحالات الواردة من المرضى، فحص صور الأشعة بدقة، والتحقق من خرائط التمركز والتقارير الذكية.")

#     # محاكاة قائمة مرضى قادمين للنظام
#     patient_tab1, patient_tab2 = st.tabs(["📁 الحالة الحالية (المفحوصة)", "📊 إحصائيات النظام"])

#     with patient_tab1:
#         if 'last_result' in st.session_state:
#             st.success("تم استلام أحدث حالة مرفوعة من بوابة المرضى بنجاح.")

#             col_d1, col_d2 = st.columns(2)
#             with col_d1:
#                 st.metric("التشخيص الآلي المبدئي", st.session_state['last_result'])
#                 st.metric("نسبة الثقة التقنية", f"{st.session_state['last_conf']:.2f}%")
#             with col_d2:
#                 st.warning("⚠️ ملاحظة للطبيب: يرجى مراجعة خريطة Grad-CAM والتأكد من توافقها السريري مع العلامات الحيوية للمريض قبل الاعتماد النهائي.")

#             st.subheader("📝 التقرير الطبي المقترح بواسطة الذكاء الاصطناعي:")
#             editable_report = st.text_area("تعديل أو اعتماد التقرير الطبي:",
#                                            value=st.session_state['last_report'], height=250)

#             if st.button("✅ اعتماد التقرير وإرساله لملف المريض"):
#                 st.success("تم اعتماد التقرير الطبي بنجاح وإغلاق دورة الحالة!")
#         else:
#             st.info("لا توجد حالات جديدة معلقة حالياً. يطلب من المرضى رفع صور عبر بوابة المرضى.")

#     with patient_tab2:
#         st.subheader("📈 إحصائيات النظام السريري")
#         st.write("عدد الحالات المفحوصة اليوم: 12 حالة")
#         st.write("نسبة دقة التوافق مع التشخيص البشري: 94.2%")

# # تنبيه أكاديمي
# st.markdown("---")
# st.caption("⚠️ **تنبيه إخلاء مسؤولية أكاديمي:** هذا النظام هو نموذج أولي لمشروع تخرج جامعي (Clinical Decision Support Prototype) وليس بديلاً عن التشخيص الطبي البشري المتخصص.")









###########################
# import streamlit as st
# from PIL import Image
# import torch
# from src.model import load_model, get_transforms, GradCAM, generate_heatmap_image
# from src.llm_helper import generate_medical_report

# # إعدادات الصفحة
# st.set_page_config(page_title="MediAI - Chest Diagnostics Platform", page_icon="🏥", layout="wide")

# # مسار الموديل
# MODEL_PATH = "models/efficientnet_pneumonia.pth"

# @st.cache_resource
# def get_cached_model():
#     return load_model(MODEL_PATH)

# model = get_cached_model()

# # شريط جانبى لتحديد الدور (Dashboard Selector)
# st.sidebar.title("🏥 نظام تشخيص الصدر الذكي")
# role = st.sidebar.selectbox("اختر لوحة التحكم (Role):", ["المريض (Patient Dashboard)", "الطبيب (Doctor Dashboard)"])

# st.sidebar.markdown("---")
# st.sidebar.info("💡 مشروع تخرج - نظام مساعد ذكي لتحليل أشعة الصدر وتقديم التقارير الطبية.")

# # ==========================================
# # 1. لوحة تحكم المريض (Patient Dashboard)
# # ==========================================
# if role == "المريض (Patient Dashboard)":
#     st.title("👤 لوحة تحكم المريض - فحص الأشعة والأعراض")
#     st.write("أدخل بياناتك الصحية واقترح صورة الأشعة للحصول على تحليل مبدئي وتقرير شامل.")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📋 البيانات الشخصية والأعراض")
#         age = st.number_input("السن (العمر):", min_value=1, max_value=120, value=25)
#         gender = st.selectbox("النوع:", ["ذكر", "أنثى"])
#         symptoms = st.text_area("ما هي الأعراض التي تعاني منها؟ (مثلاً: كحة جافة، ضيق تنفس، ارتفاع حرارة...)", placeholder="اكتب الأعراض بالتفصيل...")
        
#         uploaded_file = st.file_uploader("اختر صورة أشعة الصدر (Chest X-Ray - JPG/PNG)...", type=["jpg", "png", "jpeg"])

#     with col2:
#         st.subheader("🔍 معاينة الأشعة والنتيجة")
#         if uploaded_file is not None:
#             original_image = Image.open(uploaded_file).convert('RGB')
#             st.image(original_image, caption="صورة الأشعة المرفوعة", use_column_width=True)
            
#             if st.button("🚀 ابدأ التحليل الشامل وتوليد التقرير"):
#                 with st.spinner("جاري معالجة الصورة وتحليل الحالة بالذكاء الاصطناعي..."):
                    
#                     # معالجة الصورة والتنبؤ بالموديل
#                     transform = get_transforms()
#                     input_tensor = transform(original_image).unsqueeze(0)
                    
#                     with torch.no_grad():
#                         output = model(input_tensor)
#                         prob = torch.sigmoid(output).item()
                        
#                     if prob >= 0.65:
#                         result_label = "PNEUMONIA (التهاب رئوي)"
#                         conf_score = prob * 100
#                     elif prob <= 0.35:
#                         result_label = "NORMAL (سليم)"
#                         conf_score = (1 - prob) * 100
#                     else:
#                         result_label = "BORDERLINE / UNCERTAIN تحتاج مراجعه الطبيب"
#                         conf_score =  prob * 100
                        
#                     # توليد Grad-CAM
#                     target_layer = model.features[-1]
#                     grad_cam = GradCAM(model, target_layer)
#                     cam = grad_cam(input_tensor)
#                     heatmap_img = generate_heatmap_image(original_image, cam)
                    
#                     st.image(heatmap_img, caption="Grad-CAM Heatmap (موضع التركيز للالتهاب)", use_column_width=True)
                    
#                     # حفظ النتائج في Session State لعرضها
#                     st.session_state['last_result'] = result_label
#                     st.session_state['last_conf'] = conf_score
#                     st.session_state['last_report'] = generate_medical_report(age, gender, symptoms, result_label, conf_score)

#     # عرض التقرير الطبي الناتج
#     if 'last_report' in st.session_state:
#         st.markdown("---")
#         st.subheader("📄 التقرير الطبي المبدئي وتوصيات العلاج والراحة:")
#         st.markdown(st.session_state['last_report'])

# # ==========================================
# # 2. لوحة تحكم الطبيب (Doctor Dashboard)
# # ==========================================
# elif role == "الطبيب (Doctor Dashboard)":
#     st.title("🩺 لوحة تحكم الطبيب - مراجعة الحالات واعتماد التقارير")
#     st.write("هنا يتم مراجعة الحالات الواردة من المرضى، فحص صور الأشعة بدقة، والتحقق من خرائط التمركز والتقارير الذكية.")
    
#     # محاكاة قائمة مرضى قادمين للنظام
#     patient_tab1, patient_tab2 = st.tabs(["📁 الحالة الحالية (المفحوصة)", "📊 إحصائيات النظام"])
    
#     with patient_tab1:
#         if 'last_result' in st.session_state:
#             st.success("تم استلام أحدث حالة مرفوعة من بوابة المرضى بنجاح.")
            
#             col_d1, col_d2 = st.columns(2)
#             with col_d1:
#                 st.metric("التشخيص الآلي المبدئي", st.session_state['last_result'])
#                 st.metric("نسبة الثقة التقنية", f"{st.session_state['last_conf']:.2f}%")
#             with col_d2:
#                 st.warning("⚠️ ملاحظة للطبيب: يرجى مراجعة خريطة Grad-CAM والتأكد من توافقها السريري مع العلامات الحيوية للمريض قبل الاعتماد النهائي.")
                
#             st.subheader("📝 التقرير الطبي المقترح بواسطة الذكاء الاصطناعي:")
#             editable_report = st.text_area("تعديل أو اعتماد التقرير الطبي:", value=st.session_state['last_report'], height=250)
            
#             if st.button("✅ اعتماد التقرير وإرساله لملف المريض"):
#                 st.success("تم اعتماد التقرير الطبي بنجاح وإغلاق دورة الحالة!")
#         else:
#             st.info("لا توجد حالات جديدة معلقة حالياً. يطلب من المرضى رفع صور عبر بوابة المرضى.")
            
#     with patient_tab2:
#         st.subheader("📈 إحصائيات النظام السريري")
#         st.write("عدد الحالات المفحوصة اليوم: 12 حالة")
#         st.write("نسبة دقة التوافق مع التشخيص البشري: 94.2%")

# # تنبيه أكاديمي
# st.markdown("---")
# st.caption("⚠️ **تنبيه إخلاء مسؤولية أكاديمي:** هذا النظام هو نموذج أولي لمشروع تخرج جامعي (Clinical Decision Support Prototype) وليس بديلاً عن التشخيص الطبي البشري المتخصص.")




#######################
# import streamlit as st
# from PIL import Image
# import torch
# from src.model import load_model, get_transforms, GradCAM, generate_heatmap_image

# # إعدادات صفحة الستريملت
# st.set_page_config(page_title="Chest X-Ray Pneumonia Diagnosis", page_icon="🫁", layout="centered")

# st.title("🫁 Chest X-Ray Pneumonia Classification")
# st.write("ارفع صورة الأشعة السينية للصدر (Chest X-Ray) ليكتشف الذكاء الاصطناعي ما إذا كانت سليمة أم مصابة بالالتهاب الرئوي، مع توضيح مكان الإصابة بالخريطة الحرارية (Grad-CAM).")

# # مسار ملف الأوزان
# MODEL_PATH = "models/efficientnet_pneumonia.pth"

# @st.cache_resource
# def get_cached_model():
#     return load_model(MODEL_PATH)

# with st.spinner("جاري تحميل الموديل... يرجى الانتظار"):
#     model = get_cached_model()

# # رفع الصورة
# uploaded_file = st.file_uploader("اختر صورة أشعة (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # فتح الصورة الأصلية
#     original_image = Image.open(uploaded_file).convert('RGB')
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(original_image, caption="صورة الأشعة المرفوعة", use_column_width=True)
    
#     if st.button("🚀 ابدأ تحليل الصورة وتوليد التقرير"):
#         with st.spinner("جاري المعالجة وتحليل الأنسجة بالذكاء الاصطناعي..."):
            
#             # 1. تجهيز الصورة للموديل
#             transform = get_transforms()
#             input_tensor = transform(original_image).unsqueeze(0) # إضافة دفعة (Batch Dimension)
            
#             # 2. التنبؤ (Inference) باستخدام Sigmoid لأن الخروج بقيمة واحدة
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 prob = torch.sigmoid(output).item() # النسبة بين 0 و 1
                
#             # تحديد الفئة بناءً على عتبة 0.5
#             if prob >= 0.5:
#                 result_label = "PNEUMONIA"
#                 conf_score = prob * 100
#             else:
#                 result_label = "NORMAL"
#                 conf_score = (1 - prob) * 100
                
#             # 3. توليد Grad-CAM للطبقة الأخيرة في EfficientNet (`model.features[-1]`)
#             target_layer = model.features[-1] 
#             grad_cam = GradCAM(model, target_layer)
#             cam = grad_cam(input_tensor)
            
#             # 4. دمج الـ Heatmap مع الصورة
#             heatmap_img = generate_heatmap_image(original_image, cam)
            
#             with col2:
#                 st.image(heatmap_img, caption="Grad-CAM Heatmap (موضع التركيز)", use_column_width=True)
            
#             # 5. عرض النتيجة النهائية للمستخدم
#             st.markdown("---")
#             st.subheader("📋 نتيجة التشخيص الطبي:")
#             if result_label == "PNEUMONIA":
#                 st.error(f"**التشخيص المتوقع:** `{result_label}` ⚠️")
#             else:
#                 st.success(f"**التشخيص المتوقع:** `{result_label}` ✅")
                
#             st.metric(label="نسبة الثقة (Confidence Score)", value=f"{conf_score:.2f}%")
#             st.info("💡 **ملاحظة توضيحية:** الخريطة الحرارية (Grad-CAM) توضح بالألوان المناطق التي اعتمد عليها الموديل بشكل أساسي لاتخاذ قراره التشخيصي.")




###################
# import streamlit as st
# from PIL import Image
# import torch
# from src.model import load_model, get_transforms, GradCAM, generate_heatmap_image

# # إعدادات الصفحة
# st.set_page_config(page_title="Chest X-Ray Pneumonia Diagnosis", page_icon="🫁", layout="centered")

# st.title("🫁 Chest X-Ray Pneumonia Classification")
# st.write("ارفع صورة الأشعة السينية للصدر لفحصها واكتشاف الالتهاب الرئوي مع توضيح مكان الإصابة بالخريطة الحرارية.")

# # مسار ملف أوزان الموديل (تأكد من اسم ومسار الملف الصحيح عندك)
# MODEL_PATH = "models/efficientnet_pneumonia.pth" # عدل المسار لاسم ملف الـ pth الفعلي عندك

# @st.cache_resource
# def get_cached_model():
#     return load_model(MODEL_PATH)

# with st.spinner("جاري تحميل الموديل..."):
#     model = get_cached_model()

# # رفع الصورة
# uploaded_file = st.file_uploader("اختر صورة أشعة (JPG, PNG)...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     original_image = Image.open(uploaded_file).convert('RGB')
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(original_image, caption="صورة الأشعة المرفوعة", use_column_width=True)
    
#     if st.button("🚀 ابدأ التحليل"):
#         with st.spinner("جاري المعالجة والتحليل بالذكاء الاصطناعي..."):
            
#             # تجهيز الصورة والتنبؤ
#             transform = get_transforms()
#             input_tensor = transform(original_image).unsqueeze(0)
            
#             with torch.no_grad():
#                 outputs = model(input_tensor)
#                 probabilities = torch.nn.functional.softmax(outputs, dim=1)
#                 confidence, predicted_class = torch.max(probabilities, 1)
                
#             classes = ["NORMAL", "PNEUMONIA"]
#             result_label = classes[predicted_class.item()]
#             conf_score = confidence.item() * 100
            
#             # توليد Grad-CAM (لو موديلك DenseNet استخدم model.features[-1])
#             target_layer = model.features[-1] 
#             grad_cam = GradCAM(model, target_layer)
#             cam, _ = grad_cam(input_tensor, predicted_class.item())
            
#             heatmap_img = generate_heatmap_image(original_image, cam)
            
#             with col2:
#                 st.image(heatmap_img, caption="Grad-CAM Heatmap (موضع التركيز)", use_column_width=True)
            
#             st.markdown("---")
#             st.subheader("📋 نتيجة التشخيص الطبي:")
#             if result_label == "PNEUMONIA":
#                 st.error(f"**التشخيص المتوقع:** `{result_label}` ⚠️")
#             else:
#                 st.success(f"**التشخيص المتوقع:** `{result_label}` ✅")
                
#             st.metric(label="نسبة الثقة (Confidence Score)", value=f"{conf_score:.2f}%")



####################
# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms
# from src.model import EfficientNetPneumonia, generate_gradcam_efficientnet
# import cv2
# import numpy as np

# st.set_page_config(page_title="Pneumonia AI Detection", layout="wide")

# st.title("🩺 Chest X-Ray Diagnosis with EfficientNetB0 & Grad-CAM")

# # شريط تحكم الـ Threshold في الجانب
# threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

# uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption="Uploaded X-Ray", width=300)

#     # معالجة الصورة (Resize 224x224 والتنسيق)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])
#     tensor = transform(image).unsqueeze(0)

#     # تحميل الموديل والأوزان من مجلد models
#     model = EfficientNetPneumonia()
#     model.load_state_dict(torch.load("models/efficientnet_pneumonia.pth", map_location='cpu'))
#     model.eval()

#     with torch.no_grad():
#         prob = model(tensor).item()

#     result = "PNEUMONIA DETECTED" if prob >= threshold else "NORMAL"

#     if result.startswith("PNEUMONIA"):
#         st.error(f"Prediction: {result} (Probability: {prob:.2f})")
#     else:
#         st.success(f"Prediction: {result} (Probability: {1-prob:.2f})")

#     # Grad-CAM Heatmap
#     if st.checkbox("Show Grad-CAM Heatmap"):
#         heatmap = generate_gradcam_efficientnet(model, tensor)
#         img_np = np.array(image.resize((224, 224)))
#         heatmap_resized = cv2.resize(heatmap, (224, 224))
#         heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
#         superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
#         st.image(superimposed, caption="Grad-CAM Focus Area", width=300)