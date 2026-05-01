import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="AgroScan AI",
    page_icon="leaf",
    layout="wide",
    initial_sidebar_state="expanded"
)

TIPS = {
    "banana_bract_mosaic_virus"       : "Remove and destroy infected plants immediately. Use virus-free suckers for replanting. Control aphid vectors with neem oil spray.",
    "banana_cordana"                  : "Remove lower infected leaves. Apply copper oxychloride fungicide. Ensure good drainage around plant base.",
    "banana_healthy"                  : "Plant looks healthy! Maintain regular watering, balanced fertilisation, and monitor weekly.",
    "banana_insectpest"               : "Apply neem-based insecticide or imidacloprid. Remove heavily infested leaves. Use sticky traps near plants.",
    "banana_moko"                     : "No cure available — remove and destroy infected plants. Disinfect tools with bleach. Avoid moving soil from infected areas.",
    "banana_panama"                   : "No cure — remove infected plants. Improve soil drainage. Plant resistant varieties in future.",
    "banana_pestalotiopsis"           : "Remove infected leaves. Apply mancozeb or copper fungicide. Avoid water stress.",
    "banana_sigatoka"                 : "Remove and destroy infected leaves. Apply copper-based or mancozeb fungicide every 2 weeks. Improve air circulation.",
    "banana_yb_sigatoka"              : "Apply propiconazole or trifloxystrobin fungicide. Remove severely infected leaves. Avoid overhead irrigation.",
    "cauliflower_Blackrot"            : "Use certified pathogen-free seeds. Apply copper bactericide. Practice 3-year crop rotation with non-brassicas.",
    "cauliflower_bacterial _spot _rot": "Avoid overhead irrigation. Apply copper-based bactericide. Remove and destroy infected plant debris.",
    "cauliflower_downy_mildew"        : "Apply metalaxyl-M or fosetyl-aluminium fungicide. Ensure wide plant spacing. Avoid wetting foliage.",
    "cauliflower_healthy"             : "Plant looks healthy! Maintain consistent moisture and nutrition for optimal curd development.",
    "chilli_anthracnose"              : "Avoid overhead irrigation. Spray mancozeb or copper fungicide. Remove and destroy infected fruits immediately.",
    "chilli_healthy"                  : "Plant looks healthy! Maintain proper spacing for good air circulation and scout regularly.",
    "chilli_leafcurl"                 : "Control whitefly vectors with imidacloprid or neem oil. Remove severely infected plants. Use resistant varieties.",
    "chilli_leafspot"                 : "Apply chlorothalonil or mancozeb fungicide. Avoid overhead irrigation. Remove infected leaves promptly.",
    "chilli_whitefly"                 : "Apply imidacloprid or neem oil spray. Use yellow sticky traps. Remove heavily infested leaves.",
    "chilli_yellowish"                : "Check soil nutrition — likely nitrogen or magnesium deficiency. Apply balanced fertiliser. Ensure proper irrigation.",
    "groundnut_early_leaf_spot"       : "Spray chlorothalonil or mancozeb every 10-14 days. Rotate with non-host crops. Remove crop debris after harvest.",
    "groundnut_early_rust"            : "Apply propiconazole or mancozeb fungicide at first sign. Maintain proper plant spacing. Remove volunteer plants.",
    "groundnut_healthy"               : "Plant looks healthy! Scout regularly and maintain good soil drainage and adequate spacing.",
    "groundnut_late_leaf_spot"        : "Apply carbendazim + mancozeb combination spray. Practice crop rotation. Destroy infected plant debris.",
    "groundnut_nutrition_deficiency"  : "Conduct soil test and apply recommended fertiliser. Apply micronutrient mix including zinc and boron.",
    "groundnut_rust"                  : "Apply propiconazole fungicide early. Maintain proper spacing. Remove and destroy heavily infected plants.",
    "radish_black_leaf_spot"          : "Apply copper-based fungicide. Avoid overhead irrigation. Remove infected leaves and improve air circulation.",
    "radish_downey_mildew"            : "Apply metalaxyl fungicide. Improve air circulation. Avoid wetting foliage during irrigation.",
    "radish_flea_beetle"              : "Apply carbaryl or spinosad insecticide. Use row covers for protection. Remove crop debris after harvest.",
    "radish_healthy"                  : "Plant looks healthy! Ensure consistent watering and avoid waterlogging to maintain root quality.",
    "radish_mosaic"                   : "Remove and destroy infected plants. Control aphid vectors with neem oil. Use certified virus-free seeds.",
}
DEFAULT_TIP = "Consult a local agronomist or plant pathology lab for a tailored treatment recommendation."


@st.cache_resource
def load_model():
    model_path   = "plant_disease_model.h5"
    classes_path = "class_names.json"
    if not os.path.exists(model_path):
        return None, None
    m = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        c = json.load(f)
    return m, c


model, CLASS_NAMES = load_model()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AgroScan AI")
    st.caption("Multi-Crop Disease & Health Detector")
    st.divider()
    st.markdown("**Supported crops**")
    for crop in ["Banana", "Chilli", "Radish",
                 "Groundnut", "Cauliflower"]:
        st.markdown(f"- {crop}")
    st.divider()
    st.markdown("**Model info**")
    st.caption("Architecture  : 3-Block Vanilla CNN")
    st.caption("Optimizer     : SGD + Momentum")
    st.caption("Regularisation: Dropout + L2")
    st.caption("Classes       : 30 disease & healthy")
    st.divider()
    st.caption("Dataset: Multi-Crop Disease Dataset")
    st.caption("Source : Mendeley Data | CC BY 4.0")
    st.caption("Images : 23,000+ leaf photos")

# ── Main page ─────────────────────────────────────────────
st.title("Multi-Crop Disease and Health Detection")
st.markdown(
    "Upload a leaf photo from any of the 5 supported crops "
    "to get an instant AI-powered disease diagnosis.")
st.divider()

if model is None:
    st.error(
        "Model not found. Place plant_disease_model.h5 "
        "and class_names.json in the same folder as app.py.")
    st.stop()

uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, well-lit photo of a single leaf")

if not uploaded:
    st.info("Upload a leaf image above to get started.")
    st.stop()

# ── Preprocess & predict ──────────────────────────────────
img = Image.open(uploaded).convert("RGB")
arr = np.array(img.resize((128, 128)),
               dtype=np.float32) / 255.0

with st.spinner("Analysing leaf..."):
    preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]

top3  = np.argsort(preds)[::-1][:3]
top_c = CLASS_NAMES[top3[0]]
top_p = float(preds[top3[0]]) * 100
parts = top_c.replace("_", " ").split()
crop  = parts[0].title()
dis   = " ".join(parts[1:]).title() if len(parts) > 1 else "Healthy"

# ── Layout ────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("**Uploaded image**")
    st.image(img, use_column_width=True, caption=uploaded.name)
    st.caption(
        f"{img.size[0]} x {img.size[1]} px  |  "
        f"{round(uploaded.size/1024, 1)} KB")

with col2:
    st.markdown("**Prediction result**")

    if top_p >= 80:
        bg, tc, bc = "#E1F5EE", "#085041", "#1D9E75"
    elif top_p >= 50:
        bg, tc, bc = "#FFF8E1", "#6D4C00", "#FFC107"
    else:
        bg, tc, bc = "#FCEBEB", "#501313", "#E24B4A"

    st.markdown(
        f"""<div style="background:{bg};border-radius:10px;
        padding:14px 18px;border-left:4px solid {bc};
        margin-bottom:14px;">
        <div style="font-size:0.8rem;color:{tc};opacity:0.8;">
            Detected condition
        </div>
        <div style="font-size:1.2rem;font-weight:600;color:{tc};">
            {top_c.replace("_", " ").title()}
        </div>
        <div style="font-size:0.9rem;color:{tc};opacity:0.85;">
            {top_p:.1f}% confidence
        </div></div>""",
        unsafe_allow_html=True)

    st.markdown("**Top 3 predictions**")
    bar_colors = ["#1D9E75", "#B4B2A9", "#D3D1C7"]
    for rank, idx in enumerate(top3):
        p  = float(preds[idx]) * 100
        bc = bar_colors[rank]
        st.markdown(
            f"""<div style="display:flex;align-items:center;
            gap:10px;margin-bottom:8px;">
            <span style="font-size:0.75rem;color:#aaa;
                min-width:16px;">{rank+1}</span>
            <span style="font-size:0.82rem;flex:1;">
                {CLASS_NAMES[idx].replace("_"," ").title()}</span>
            <div style="flex:1.2;height:7px;background:#e8e8e4;
                border-radius:4px;overflow:hidden;">
            <div style="width:{max(p,1):.0f}%;height:100%;
                border-radius:4px;background:{bc};"></div></div>
            <span style="font-size:0.8rem;color:#888;
                min-width:40px;text-align:right;">
                {p:.1f}%</span></div>""",
            unsafe_allow_html=True)

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Crop",       crop)
    m2.metric("Condition",  dis)
    m3.metric("Confidence", f"{top_p:.1f}%")

    tip = TIPS.get(top_c, DEFAULT_TIP)
    st.markdown(
        f"""<div style="background:#FFF8E1;
        border-left:3px solid #FFC107;
        border-radius:0 8px 8px 0;
        padding:10px 14px;margin-top:8px;
        font-size:0.87rem;color:#6D4C00;line-height:1.5;">
        <strong>Treatment tip:</strong> {tip}</div>""",
        unsafe_allow_html=True)

# ── Full confidence table ─────────────────────────────────
st.divider()
with st.expander("View full confidence scores for all 30 classes"):
    import pandas as pd
    idx_s = np.argsort(preds)[::-1]
    df = pd.DataFrame({
        "Class": [CLASS_NAMES[i].replace("_"," ").title()
                  for i in idx_s],
        "Confidence (%)": [round(float(preds[i])*100, 2)
                           for i in idx_s]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Class")["Confidence (%)"])

st.divider()
st.caption(
    "Built with TensorFlow · Streamlit · "
    "Multi-Crop Disease Dataset (Mendeley Data, CC BY 4.0) · "
    "4th Semester AIML Project")
