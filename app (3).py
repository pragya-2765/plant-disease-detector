import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="🧑‍🌾 Crop Intelligence", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

/* Background */
.stApp {
    background: linear-gradient(135deg, #f7fbe8, #edf7c5);
}

/* Floating leaves (more density) */
@keyframes float {
    0% { transform: translateY(100vh); opacity: 0; }
    50% { opacity: 0.7; }
    100% { transform: translateY(-10vh); opacity: 0; }
}


.leaf {
    position: fixed;
    font-size: 22px;
    animation: float 10s linear infinite;
}

/* Generate many leaves */
.leaf:nth-child(1){left:5%;animation-delay:0s;}
.leaf:nth-child(2){left:15%;animation-delay:2s;}
.leaf:nth-child(3){left:25%;animation-delay:4s;}
.leaf:nth-child(4){left:35%;animation-delay:1s;}
.leaf:nth-child(5){left:45%;animation-delay:3s;}
.leaf:nth-child(6){left:55%;animation-delay:5s;}
.leaf:nth-child(7){left:65%;animation-delay:2s;}
.leaf:nth-child(8){left:75%;animation-delay:6s;}
.leaf:nth-child(9){left:85%;animation-delay:1s;}
.leaf:nth-child(10){left:95%;animation-delay:4s;}

/* Title */
.title {
    font-size: 60px;
    font-weight: 800;
    color: #1b5e20;
}

.mid-title {
    font-size: 28px;
    font-weight: 600;
    color: #558b2f;
    margin-top: 10px;
    margin-bottom: 8px;
}

/* Container to stack elements */
.upload-container {
    position: relative;
    width: 100%;
    height: 280px;
}

/* Actual uploader (invisible but clickable) */
.upload-container input[type="file"] {
    position: absolute;
    width: 100%;
    height: 100%;
    opacity: 0;
    cursor: pointer;
    z-index: 2;
}

/* Custom UI box */
.upload-box {
    border: 2px dashed #4caf50;
    border-radius: 18px;
    padding: 30px;
    min-height: 280px;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-direction:column;
    background:#f1f8e9;
}

.upload-box:hover {
    background: #e8f5e9;
}


/* Image container */
.image-box {
    width: 100%;
    height: 280px;
    border-radius: 18px;
    overflow: hidden;
    border: 2px dashed #4caf50;
}

/* Image inside box */
.image-box img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover;
    border-radius: 18px;
}

/* Cross button */
.cross {
    position:absolute;
    top:8px;
    right:10px;
    font-size:20px;
    cursor:pointer;
    background:white;
    border-radius:50%;
    padding:2px 6px;
}

/* Predict button */
.stButton>button {
    background: linear-gradient(to right, #ffeb3b, #ffc107);
    color:black;
    font-weight:bold;
    font-size:18px;
    height:55px;
    border-radius:12px;
}

/* Result */
.result {
    margin-top:10px;
    background:#fff8c6;
    padding:12px;
    border-radius:10px;
    font-size:16px;
}
</style>

<!-- Leaves -->
<div class="leaf">🌿</div>
<div class="leaf">🌸</div>
<div class="leaf">🍃</div>
<div class="leaf">🌼</div>
<div class="leaf">🍂</div>
<div class="leaf">🌾</div>
<div class="leaf">🌱</div>
<div class="leaf">🍁</div>
<div class="leaf">🍀</div>
<div class="leaf">🪻</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

with open('class_names.json') as f:
    class_names = json.load(f)

# ---------------- LAYOUT ----------------
# ---------------- LAYOUT ----------------
left, right = st.columns([2,1])

# LEFT SIDE (TEXT HEAVY)
with left:
    st.markdown('<div class="title">🧑‍🌾 Crop Disease Detector</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:26px; font-weight:600; color:#558b2f;">
    Detect plant disease from a leaf image 🌿
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-size:18px; color:#444; margin-top:10px; line-height:1.6;">From leaf to logic: Bridging the gap between a captured image and a saved crop.<br><br></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:18px; color:#444; margin-top:10px; line-height:1.6;">
    Upload a crop leaf image and instantly identify diseases using deep learning.<br><br>
    This tool helps farmers and researchers detect infections early, 
    improve crop yield, and make smarter agricultural decisions.
    <br><br>
    Simply upload an image → click predict → get accurate results within seconds.
    </div>
    """, unsafe_allow_html=True)


# RIGHT SIDE (UPLOAD + RESULT)
with right:

    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Image",
        type=["jpg","png","jpeg"]
    )

    predict_clicked = False

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, use_container_width=True)

        predict_clicked = st.button("🌿 Predict", use_container_width=True)


# ✅ FULL-WIDTH RESULT (outside right column)
if uploaded_file is not None and predict_clicked:

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    # Top prediction
    top_index = np.argmax(prediction)
    top_class = class_names[top_index]
    top_conf = prediction[top_index]

    # Top 3 predictions
    top3_idx = prediction.argsort()[-3:][::-1]
    top3 = [(class_names[i], prediction[i], i) for i in top3_idx]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🌿 Prediction Results")

    col1, col2, col3 = st.columns(3)

    # -------- TOP PREDICTION --------
    with col1:
      st.markdown(f"""
      <div style="
          background:white;
          padding:20px;
          border-radius:16px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.08);
          height:180px;
      ">
      
      <div style="font-size:14px; color:#2e7d32; font-weight:600;">
      Top prediction
      </div>

      <div style="
          font-size:24px;
          font-weight:700;
          color:#1b5e20;
          margin-top:8px;
          white-space:nowrap;
          overflow:hidden;
          text-overflow:ellipsis;
      ">
      {top_class.replace("_"," ")}
      </div>

      <div style="margin-top:10px; font-size:14px; color:#2e7d32;">
      Confidence: <b>{top_conf*100:.2f}%</b><br>
      Class index: {top_index}
      </div>

      </div>
      """, unsafe_allow_html=True)

    # -------- TOP 3 --------
    with col2:
      html = """
      <div style="
      background:white;
      padding:20px;
      border-radius:16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      color:#1b5e20;
      height:180px;
      overflow:auto;
      ">
      <b style="color:#2e7d32;">Top 3 predictions</b><br><br>
      """

      for name, conf, idx in top3:
          html += f"{name.replace('_',' ')}<br><span style='color:#2e7d32;'>Class {idx} — {conf*100:.2f}%</span><br><br>"

      html += "</div>"

      st.markdown(html, unsafe_allow_html=True)

    # -------- MODEL METRICS --------
    with col3:
      st.markdown(f"""
      <div style="
          background:white;
          padding:20px;
          border-radius:16px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.08);
          color:#1b5e20;
          height:180px;
      ">
      <b style="color:#2e7d32;">Model metrics</b><br><br>
      Classes: {len(class_names)}<br>
      Probability sum: {np.sum(prediction):.6f}<br>
      Input shape: 224 x 224 x 3
      </div>
      """, unsafe_allow_html=True)
