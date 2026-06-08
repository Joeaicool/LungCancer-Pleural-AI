import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import tempfile
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from radiomics import featureextractor
import urllib.request

# =========================
# 1. 页面配置与高级医学 CSS
# =========================
st.set_page_config(page_title="Pleural Invasion Predictor | Lung Cancer AI", page_icon="🫁", layout="wide")

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .title-box { background: linear-gradient(135deg, #0A2540 0%, #1750A1 100%); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .title-box h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
    .title-box p { margin-top: 10px; font-size: 1.1rem; opacity: 0.9; }
    .card { background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 1.5rem; }
    .card-title { color: #0A2540; font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; border-bottom: 2px solid #F1F5F9; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1>Multi-modal AI Framework for Predicting Pleural Invasion in Lung Cancer</h1>
    <p>Integrating 3D Deep Learning, Radiomics, and Clinical Biomarkers</p>
</div>
""", unsafe_allow_html=True)

# =========================
# 加载题图 (lc.png)
# =========================
try:
    from PIL import Image
    if os.path.exists("lc.png"):
        img = Image.open("lc.png")
        st.image(img, use_container_width=True)
except Exception:
    pass

# =========================
# 2. 全局配置与特征列表
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ML_MODEL_PATH = "RF_PSO_best.pkl"
DL_WEIGHT_PATH = "resnet10.pth"
MEDICALNET_DIR = "./MedicalNet"

ALL_FEATURES = [
    "wavelet-LLL_gldm_LargeDependenceHighGrayLevelEmphasis",
    "log-sigma-3-0-mm-3D_firstorder_Range",
    "log-sigma-3-0-mm-3D_glszm_ZoneEntropy",
    "wavelet-LLL_gldm_SmallDependenceLowGrayLevelEmphasis",
    "wavelet-LLL_glszm_SmallAreaLowGrayLevelEmphasis",
    "original_firstorder_RootMeanSquared",
    "log-sigma-3-0-mm-3D_glszm_GrayLevelVariance",
    "CEA",
    "log-sigma-3-0-mm-3D_gldm_DependenceEntropy",
    "wavelet-LLL_glszm_SmallAreaHighGrayLevelEmphasis",
    "wavelet-LLL_firstorder_10Percentile",
    "wavelet-LLH_firstorder_Maximum",
    "DL_feat_0005",
    "CA125",
    "original_glrlm_RunEntropy",
    "DL_feat_0012",
    "wavelet-LHH_gldm_LargeDependenceHighGrayLevelEmphasis",
    "wavelet-LLL_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis",
    "age"
]

CLINICAL_FEATS = ["CEA", "CA125", "age"]
DL_FEATS = ["DL_feat_0005", "DL_feat_0012"]
RAD_FEATS = [f for f in ALL_FEATURES if f not in CLINICAL_FEATS and f not in DL_FEATS]

# =========================
# 3. 核心工具函数：图像预处理
# =========================
def window_clip_array(arr, level=40, width=400):
    arr = arr.astype(np.float32)
    low = level - width / 2.0
    high = level + width / 2.0
    arr = np.clip(arr, low, high)
    arr = (arr - low) / max(high - low, 1e-6)
    return arr * 2.0 - 1.0

def crop_roi_by_mask(image_arr, mask_arr, margin=8):
    coords = np.argwhere(mask_arr > 0)
    if coords.shape[0] == 0: return image_arr, mask_arr
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    D, H, W = image_arr.shape
    zmin, ymin, xmin = max(zmin-margin, 0), max(ymin-margin, 0), max(xmin-margin, 0)
    zmax, ymax, xmax = min(zmax+margin+1, D), min(ymax+margin+1, H), min(xmax+margin+1, W)
    return image_arr[zmin:zmax, ymin:ymax, xmin:xmax], mask_arr[zmin:zmax, ymin:ymax, xmin:xmax]

# =========================
# 4. 核心工具函数：模型加载与特征提取
# =========================
@st.cache_resource
def load_ml_model():
    return joblib.load(ML_MODEL_PATH)

@st.cache_resource
def load_dl_model():
    sys.path.insert(0, MEDICALNET_DIR)
    from models import resnet
    
    # 🌟 新增：自动从 GitHub Releases 下载 160MB 权重文件
    MODEL_URL = "https://github.com/Joeaicool/LungCancer-Pleural-AI/releases/download/v1.0/resnet10.pth"
    
    # 如果文件不存在，或者大小不正常(小于10MB)，就自动下载
    if not os.path.exists(DL_WEIGHT_PATH) or os.path.getsize(DL_WEIGHT_PATH) < 10000000:
        with st.spinner("Downloading Deep Learning Weights (160MB) from GitHub... This will take ~20 seconds and only happen once!"):
            urllib.request.urlretrieve(MODEL_URL, DL_WEIGHT_PATH)
    
    class MedicalNetClassifierWrapper(nn.Module):
        def __init__(self, backbone, num_classes=2):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Linear(512, 100), nn.BatchNorm1d(100), nn.ReLU(inplace=True), nn.Linear(100, num_classes)
            )
        def forward(self, x):
            feat = self.backbone(x)[0]
            feat = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
            return self.classifier(feat)

    base_model = resnet.resnet10(sample_input_W=64, sample_input_H=64, sample_input_D=64, shortcut_type='B', no_cuda=(DEVICE=="cpu"), num_seg_classes=2)
    model = MedicalNetClassifierWrapper(base_model)
    
    ckpt = torch.load(DL_WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=False)
    model.to(DEVICE).eval()
    return model

def extract_dl_features(img_path, mask_path, model):
    image = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    
    img_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
    
    img_arr = window_clip_array(img_arr)
    img_arr, mask_arr = crop_roi_by_mask(img_arr, mask_arr)
    
    x = torch.from_numpy(img_arr[None, None, ...]).float()
    x = F.interpolate(x, size=(64, 64, 64), mode="trilinear", align_corners=False).to(DEVICE)
    
    features = {}
    def hook_fn(m, i): features["feat"] = i[0].clone().view(i[0].size(0), -1)
    
    handle = model.classifier[3].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    handle.remove()
    
    feat_vector = features["feat"][0].cpu().numpy()
    return {"DL_feat_0005": feat_vector[5], "DL_feat_0012": feat_vector[12]}

def extract_radiomics_features(img_path, mask_path):
    settings = {"binWidth": 25, "resampledPixelSpacing": [1.0, 1.0, 1.0], "interpolator": sitk.sitkBSpline, "correctMask": True, "label": 1}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("LoG", customArgs={"sigma": [3.0]})
    extractor.enableImageTypeByName("Wavelet")
    extractor.disableAllFeatures()
    for f in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm"]:
        extractor.enableFeatureClassByName(f)

    image = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    image = sitk.Clamp(sitk.Cast(image, sitk.sitkFloat32), lowerBound=-160, upperBound=240)
    mask = sitk.BinaryThreshold(sitk.Cast(mask, sitk.sitkFloat32), lowerThreshold=0.5, upperThreshold=100, insideValue=1, outsideValue=0)
    
    result = extractor.execute(image, sitk.Cast(mask, sitk.sitkUInt8))
    
    extracted_rad = {}
    for rad_name in RAD_FEATS:
        for key, val in result.items():
            if rad_name in key:
                extracted_rad[rad_name] = float(val)
                break
        if rad_name not in extracted_rad:
            extracted_rad[rad_name] = 0.0
            
    return extracted_rad

# =========================
# 5. UI: 数据输入区
# =========================
st.markdown('<div class="card"><div class="card-title">📁 Step 1: Upload Lung CT Scans (CT & ROI)</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    ct_file = st.file_uploader("Upload Lung CT Scan (.nii.gz)", type=["nii", "gz"], key="ct")
with c2:
    roi_file = st.file_uploader("Upload Tumor ROI Mask (.nii.gz)", type=["nii", "gz"], key="roi")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">🩸 Step 2: Clinical Biomarkers</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
clin_data = {}
with col1: clin_data["CEA"] = st.number_input("CEA (ng/mL)", min_value=0.0, value=5.0)
with col2: clin_data["CA125"] = st.number_input("CA125 (U/mL)", min_value=0.0, value=35.0)
with col3: clin_data["age"] = st.number_input("Age (Years)", min_value=18, max_value=120, value=60)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 6. 推理与预测
# =========================
_, center_col, _ = st.columns([1, 2, 1])
if center_col.button("🚀 Run Automated Extraction & Predict", type="primary", use_container_width=True):
    
    if not ct_file or not roi_file:
        st.error("⚠️ Please upload both CT scan and ROI mask to proceed.")
        st.stop()
        
    try:
        with st.spinner("Saving uploaded files..."):
            temp_dir = tempfile.mkdtemp()
            ct_path = os.path.join(temp_dir, "ct.nii.gz")
            roi_path = os.path.join(temp_dir, "roi.nii.gz")
            with open(ct_path, "wb") as f: f.write(ct_file.getbuffer())
            with open(roi_path, "wb") as f: f.write(roi_file.getbuffer())

        with st.spinner("🤖 Extracting Deep Learning Features (3D-ResNet)..."):
            dl_model = load_dl_model()
            dl_features = extract_dl_features(ct_path, roi_path, dl_model)
            
        with st.spinner("🧬 Extracting Radiomics Features (PyRadiomics)..."):
            rad_features = extract_radiomics_features(ct_path, roi_path)
            
        with st.spinner("🧠 Merging Features and Predicting Pleural Invasion..."):
            ml_model = load_ml_model()
            
            final_data = {}
            final_data.update(clin_data)
            final_data.update(dl_features)
            final_data.update(rad_features)
            
            X_input = pd.DataFrame([[final_data[k] for k in ALL_FEATURES]], columns=ALL_FEATURES)
            prob_pos = ml_model.predict_proba(X_input)[0][1] * 100

        # =========================
        # 7. 结果展示区
        # =========================
        st.markdown('<div class="card"><div class="card-title">📊 Step 3: Diagnostic Results</div>', unsafe_allow_html=True)
        res_c1, res_c2 = st.columns([1.2, 1])

        with res_c1:
            st.markdown('#### 🩺 Clinical Interpretation')
            if prob_pos >= 50:
                st.error("### ⚠️ High Risk of Pleural Invasion")
                st.write("The model indicates a **higher likelihood** of visceral pleural invasion (VPI). Closer evaluation during surgery and potential upstaging consideration may be necessary.")
            else:
                st.success("### ✅ Low Risk (Intact Pleura)")
                st.write("The model indicates a **lower likelihood** of pleural invasion. The visceral pleura is likely intact.")
            st.info(f"**Calculated Probability of Invasion:** **{prob_pos:.2f}%**")
            
            with st.expander("Show Extracted Features"):
                st.json(final_data)

        with res_c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_pos, number={"suffix": "%", "font": {"size": 40}},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1750A1"},
                       "steps": [{"range": [0, 50], "color": "rgba(40, 167, 69, 0.2)"},
                                 {"range": [50, 100], "color": "rgba(220, 53, 69, 0.2)"}]}
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # 8. SHAP 解释区
        # =========================
        st.markdown('<div class="card"><div class="card-title">🔍 Step 4: AI Explainability (SHAP)</div>', unsafe_allow_html=True)
        with st.spinner('Calculating SHAP values...'):
            try:
                explainer = shap.TreeExplainer(ml_model)
                shap_values_raw = explainer.shap_values(X_input)
                
                if isinstance(shap_values_raw, list): sv_values = shap_values_raw[1][0]
                elif len(shap_values_raw.shape) == 3: sv_values = shap_values_raw[0, :, 1]
                else: sv_values = shap_values_raw[0]

                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)): base_val = base_val[1] if len(base_val)>1 else base_val[0]

                sv_in_plot = shap.Explanation(
                    values=sv_values, base_values=float(base_val), data=X_input.iloc[0].values, feature_names=ALL_FEATURES
                )

                p1, p2 = st.columns(2)
                with p1:
                    fig_wf, ax_wf = plt.subplots(figsize=(6, 5), dpi=150)
                    shap.plots.waterfall(sv_in_plot, max_display=10, show=False)
                    st.pyplot(fig_wf)
                    plt.close(fig_wf)

                with p2:
                    contrib_df = pd.DataFrame({
                        "Feature": [f[:20]+".." if len(f)>20 else f for f in ALL_FEATURES],
                        "Effect": ["⬆️ Drives towards Invasion" if v > 0 else "⬇️ Protects Pleura" for v in sv_values],
                        "Impact": np.abs(sv_values)
                    }).sort_values("Impact", ascending=False).head(10)
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"⚠️ SHAP explanation failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")
