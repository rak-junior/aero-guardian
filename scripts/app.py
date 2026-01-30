"""
AeroGuardian Demo UI
====================================
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(page_title="AeroGuardian", page_icon="🛡️", layout="wide")

# Clean CSS matching the mockup
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
}

.stApp { 
    background: #FFFFFF; 
}

#MainMenu, footer, header { visibility: hidden; }

.block-container { 
    padding: 2rem 3rem; 
    max-width: 1200px; 
}

/* Section styling */
.section {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}

.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #1F2937;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-icon {
    color: #0EA5E9;
}

/* File uploader */
.stFileUploader {
    background: #FAFAFA;
    border: 2px dashed #D1D5DB;
    border-radius: 8px;
    padding: 20px;
}

.stFileUploader:hover {
    border-color: #0EA5E9;
    background: #F0F9FF;
}

/* Button styling - Sky blue theme */
.stButton > button {
    background: #0EA5E9;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 13px;
}

.stButton > button:hover {
    background: #0284C7;
}

/* Download buttons */
.stDownloadButton > button {
    background: #0EA5E9;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
}

.stDownloadButton > button:hover {
    background: #0284C7;
}

/* Dataframe styling */
.stDataFrame {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}

/* Log area */
.log-container {
    background: #FAFAFA;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 16px;
    font-family: Monaco, 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.8;
    max-height: 250px;
    overflow-y: auto;
}

.log-info { color: #6B7280; }
.log-success { color: #059669; }
.log-warning { color: #D97706; }
.log-error { color: #DC2626; }

/* Progress */
.stProgress > div > div {
    background: #0EA5E9;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_state():
    defaults = {
        "logs": [],
        "reports": [],
        "output_paths": [],
        "all_records": [],
        "current_idx": 0,
        "total_records": 0,
        "processing": False,
        "done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def add_log(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"t": ts, "m": msg, "l": level})


def parse_file(uploaded):
    """Parse uploaded file - supports single object, arrays, nested structures.
    Returns only the FIRST record for demo purposes.
    """
    ext = Path(uploaded.name).suffix.lower()
    records = []
    
    if ext == ".csv":
        records = pd.read_csv(uploaded).to_dict(orient="records")
    elif ext == ".xlsx":
        records = pd.read_excel(uploaded).to_dict(orient="records")
    elif ext == ".json":
        data = json.load(uploaded)
        # Handle all formats: single object, array, nested
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ["incidents", "data", "records", "items", "results"]:
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            else:
                # Single object
                records = [data]
    
    # Return only first record for demo (but we parsed flexibly)
    if records:
        return records[0], len(records)  # Return first record + total count
    return None, 0



# =============================================================================
# Header
# =============================================================================

st.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 32px; padding-bottom: 16px; border-bottom: 1px solid #E5E7EB;">
    <div style="
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    ">🛡️</div>
    <div>
        <h1 style="margin: 0; font-size: 22px; font-weight: 600; color: #1F2937;">AeroGuardian</h1>
        <p style="margin: 0; font-size: 13px; color: #6B7280;">Pre-flight Safety Analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Upload Flight Data Section
# =============================================================================

st.markdown("""
<div class="section-title">
    <span class="section-icon">📤</span> Upload Flight Data
</div>
""", unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns([3, 1])

with upload_col1:
    uploaded = st.file_uploader(
        "Drag and drop files here or click to browse (CSV, JSON, XLSX)",
        type=["json", "csv", "xlsx"],
        label_visibility="collapsed"
    )

with upload_col2:
    if uploaded:
        uploaded.seek(0)
        record, total_count = parse_file(uploaded)
        if record:
            st.session_state.all_records = record  # Single record dict
            st.session_state.total_records = 1
            st.session_state.original_count = total_count

if st.session_state.total_records > 0:
    if st.session_state.get('original_count', 1) > 1:
        st.markdown(f"<div style='color: #D97706; font-size: 13px; margin-top: 8px;'>⚠️ {st.session_state.original_count} records found - using first record only</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color: #059669; font-size: 13px; margin-top: 8px;'>✓ 1 record loaded</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)


# =============================================================================
# Data Preview Section
# =============================================================================

st.markdown("""
<div class="section-title">
    <span class="section-icon">📊</span> Data Preview
</div>
""", unsafe_allow_html=True)

if st.session_state.all_records:
    # Single record - convert to DataFrame with one row
    record = st.session_state.all_records
    df = pd.DataFrame([record])
    display_cols = [c for c in ["incident_id", "city", "state", "incident_type"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=100)
else:
    st.markdown("""
    <div style="
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 40px;
        text-align: center;
        color: #9CA3AF;
    ">
        No data loaded. Upload a file to preview.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)


# =============================================================================
# Analysis Log Section
# =============================================================================

st.markdown("""
<div class="section-title">
    <span class="section-icon">📋</span> Analysis Log
</div>
""", unsafe_allow_html=True)

# Progress bar
if st.session_state.processing or st.session_state.done:
    progress = st.session_state.current_idx / st.session_state.total_records if st.session_state.total_records > 0 else 0
    st.progress(progress)
    st.markdown(f"<div style='font-size: 12px; color: #6B7280; margin-bottom: 12px;'>Progress: {st.session_state.current_idx}/{st.session_state.total_records} records</div>", unsafe_allow_html=True)

# Log display
if st.session_state.logs:
    log_html = '<div class="log-container">'
    for e in st.session_state.logs[-50:]:
        level_class = f"log-{e.get('l', 'info')}"
        prefix = {"info": "Info", "success": "✓", "warning": "Warning", "error": "Error"}.get(e.get('l'), "Info")
        log_html += f'<div class="{level_class}"><span style="color: #9CA3AF;">{e.get("t", "")}</span>  {prefix}  {e.get("m", "")}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="log-container" style="text-align: center; color: #9CA3AF; padding: 60px;">
        Analysis logs will appear here...
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)


# =============================================================================
# Download Reports Section
# =============================================================================

st.markdown("""
<div class="section-title">
    <span class="section-icon">📥</span> Download Reports
</div>
""", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.2, 1.2, 1, 2])

with btn_col1:
    # Start Analysis button
    if st.session_state.total_records > 0 and not st.session_state.processing and not st.session_state.done:
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state.processing = True
            st.session_state.current_idx = 0
            st.session_state.reports = []
            st.session_state.output_paths = []
            st.session_state.logs = []
            st.session_state.done = False
            add_log("Application started.", "info")
            add_log(f"Uploaded flight data ({st.session_state.total_records} records)", "info")
            add_log("Starting pre-flight analysis...", "info")
            st.rerun()
    elif st.session_state.processing:
        st.info("⏳ Processing...")
    elif st.session_state.done:
        st.success("✅ Complete")

with btn_col2:
    # PDF Download
    if st.session_state.reports:
        pdf_paths = [r.get("pdf_path") for r in st.session_state.reports if r.get("pdf_path") and os.path.exists(str(r.get("pdf_path", "")))]
        if pdf_paths and len(pdf_paths) == 1:
            with open(pdf_paths[0], "rb") as f:
                st.download_button("� Download PDF Report", f.read(), "report.pdf", "application/pdf", use_container_width=True)
        elif pdf_paths:
            st.download_button("� Download PDF Report", "Multiple PDFs available", "info.txt", disabled=True, use_container_width=True)
        else:
            st.button("📄 Download PDF Report", disabled=True, use_container_width=True)
    else:
        st.button("📄 Download PDF Report", disabled=True, use_container_width=True)

with btn_col3:
    # JSON Download
    if st.session_state.reports:
        count = len(st.session_state.reports)
        if count == 1:
            json_data = json.dumps(st.session_state.reports[0].get("report", {}), indent=2, default=str)
        else:
            json_data = json.dumps({
                "reports": [r.get("report", {}) for r in st.session_state.reports],
                "total": count
            }, indent=2, default=str)
        st.download_button("📊 Download JSON Report", json_data, "report.json", "application/json", use_container_width=True)
    else:
        st.button("📊 Download JSON Report", disabled=True, use_container_width=True)

with btn_col4:
    # Output folder info
    if st.session_state.output_paths:
        st.markdown(f"<div style='font-size: 11px; color: #6B7280; padding-top: 8px;'>📁 {st.session_state.output_paths[-1]}</div>", unsafe_allow_html=True)


# =============================================================================
# Full Pipeline Processing
# =============================================================================

def run_full_pipeline():
    """Run the REAL AutomatedPipeline for the single record."""
    from scripts.run_automated_pipeline import AutomatedPipeline, PipelineConfig
    
    record = st.session_state.all_records  # Single record dict
    
    incident_id = record.get("incident_id", "Upload_1")
    city = record.get("city", "Unknown")
    state = record.get("state", "")
    
    add_log(f"Analyzing: {incident_id}", "info")
    add_log(f"Location: {city}, {state}", "info")
    
    # Prepare incident
    incident = {
        "incident_id": incident_id,
        "date": record.get("date", ""),
        "city": city,
        "state": state,
        "description": record.get("description", record.get("summary", "")),
        "summary": record.get("summary", record.get("description", "")),
        "incident_type": record.get("incident_type", "other"),
    }
    
    try:
        config = PipelineConfig(headless=False)
        pipeline = AutomatedPipeline(config)
        
        add_log("Starting PX4 SITL simulation...", "info")
        add_log("This takes ~7 minutes for full flight", "warning")
        
        # Run REAL pipeline
        paths = pipeline.run_from_incident(
            incident=incident,
            skip_px4=False
        )
        
        add_log(f"Analysis complete for {incident_id}", "success")
        
        # Load report data
        report_json = paths.get("json", "")
        report_data = {}
        if report_json and os.path.exists(str(report_json)):
            with open(report_json, "r") as f:
                report_data = json.load(f)
        
        st.session_state.reports.append({
            "incident_id": incident_id,
            "report": report_data,
            "pdf_path": str(paths.get("pdf")) if paths.get("pdf") else None,
            "output_dir": str(paths.get("report_dir")) if paths.get("report_dir") else None,
        })
        
        if paths.get("report_dir"):
            st.session_state.output_paths.append(str(paths.get("report_dir")))
        
    except Exception as e:
        add_log(f"Error: {str(e)[:80]}", "error")
        import traceback
        traceback.print_exc()
    
    st.session_state.current_idx = 1
    add_log("Pipeline complete!", "success")
    st.session_state.processing = False
    st.session_state.done = True


# Run pipeline if processing
if st.session_state.processing and not st.session_state.done:
    with st.spinner("🚁 Running PX4 simulation..."):
        run_full_pipeline()
    st.rerun()


# Footer
st.markdown("<div style='height: 40px'></div>", unsafe_allow_html=True)
st.markdown("<center style='color:#9CA3AF;font-size:11px;'>AeroGuardian © 2026 | Pre-flight Safety Analysis System</center>", unsafe_allow_html=True)
