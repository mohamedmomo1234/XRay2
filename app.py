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
