"""
AeroGuardian Mission Control
====================================
Streamlined Pre-flight Safety Analysis
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="AeroGuardian", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Clean CSS Theme
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply font to text elements only, avoiding icon font breakage */
html, body, [class*="css"], font, span, div, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Status Badges */
.status-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}
.status-success { background: #Dcfce7; color: #166534; }
.status-error { background: #Fee2e2; color: #991b1b; }

/* Steps */
.step-header {
    font-size: 18px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #1e293b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

if "all_records" not in st.session_state: st.session_state.all_records = None
if "processing" not in st.session_state: st.session_state.processing = False
if "pipeline_results" not in st.session_state: st.session_state.pipeline_results = None

def parse_file(uploaded):
    """Parse file 1MB limit check handled by UI logic implicitly or uploader param."""
    if uploaded.size > 1 * 1024 * 1024:
        st.error("File size exceeds 1MB limit.")
        return None
        
    ext = Path(uploaded.name).suffix.lower()
    data = []
    try:
        if ext == ".csv":
            data = pd.read_csv(uploaded).to_dict(orient="records")
        elif ext == ".xlsx":
            data = pd.read_excel(uploaded).to_dict(orient="records")
        elif ext == ".json":
            data = json.load(uploaded)
            # Normalize to list
            if isinstance(data, dict):
                for key in ["incidents", "data", "records", "items", "results", "demo_cases", "cases", "scenarios"]:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    data = [data]
        return data
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

# =============================================================================
# Sidebar: Configuration
# =============================================================================

with st.sidebar:
    st.title("🛡️ AeroGuardian")
    st.markdown("Pre-flight Safety Analysis")
    st.markdown("---")
    
    st.markdown("**System Configuration**")
    wsl_ip = st.text_input("WSL IP Address", value="172.x.x.x", help="Run 'ip addr show eth0' in WSL to get this.")
    headless = st.toggle("Headless Mode", value=True, help="Run without Gazebo GUI (faster, less resource intensive).")
    
    st.markdown("---")
    st.info("System Ready")

# =============================================================================
# Main Content
# =============================================================================

# =============================================================================
# Main Content
# =============================================================================

st.markdown("### 1. Input Data")

# Clean vertical stack - no cramped columns
uploaded_file = st.file_uploader(
    "Upload Sighting Report (Max 1MB)", 
    type=["json", "csv", "xlsx"],
    help="Supported formats: JSON, CSV, Excel. Max size 1MB."
)

with st.expander("ℹ️  View Required Data Format & Examples"):
    st.markdown("""
    <div style="padding: 10px; background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
        <p style="margin-top:0; font-weight:600; color:#334155;">Required Fields:</p>
        <ul style="color:#475569; margin-bottom:15px;">
            <li><code>report_id</code> or <code>incident_id</code> - Unique identifier</li>
            <li><code>description</code> or <code>summary</code> - The sighting narrative text</li>
        </ul>
        <p style="margin-top:0; font-weight:600; color:#334155;">Optional Fields (Improves Simulation Accuracy):</p>
        <ul style="color:#64748b; margin-bottom:15px;">
            <li><code>date</code> - Incident date (YYYY-MM-DD or ISO format)</li>
            <li><code>city</code> - City name for location-based scenario</li>
            <li><code>state</code> - State/region for geocoding</li>
        </ul>
        <p style="font-weight:600; color:#334155;">Example JSON (FAA Format):</p>
        <pre style="background:white; padding:10px; border-radius:4px; border:1px solid #cbd5e1; font-size:12px;">
{
  "report_id": "FAA_Apr2020-Jun2020_1",
  "date": "2020-04-01",
  "city": "MINNEAPOLIS",
  "state": "MINNESOTA",
  "description": "UAS sighting reported at 3,300ft..."
}</pre>
    </div>
    """, unsafe_allow_html=True)


if uploaded_file:
    records = parse_file(uploaded_file)
    if records:
        rec = records[0]  # Take first record
        st.session_state.all_records = rec
        
        # Normalize field names - support both report_id and incident_id
        report_id = rec.get('report_id', rec.get('incident_id', 'Unknown'))
        
        # Field validation
        required_fields = ['report_id', 'description']
        optional_fields = ['date', 'city', 'state', 'summary']
        
        # Check for description/summary (accept either)
        has_description = 'description' in rec or 'summary' in rec
        has_report_id = 'report_id' in rec or 'incident_id' in rec
        
        if has_description and has_report_id:
            st.success(f"✓ Loaded: **{report_id}**")
            
            # Show field status
            found_optional = [f for f in optional_fields if f in rec]
            if found_optional:
                st.caption(f"Optional fields found: {', '.join(found_optional)}")
        else:
            missing = []
            if not has_report_id:
                missing.append('report_id')
            if not has_description:
                missing.append('description or summary')
            st.warning(f"⚠️ Missing required fields: {', '.join(missing)}")

# Step 2: Preview
if st.session_state.all_records:
    st.markdown("### 2. Data Preview")
    df = pd.DataFrame([st.session_state.all_records])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Step 3: Run Analysis
    st.markdown("### 3. Execution")
    
    # Run Button
    if st.button("🚀 Start Pipeline Analysis", type="primary", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.session_state.pipeline_results = None # Reset previous results
        
        with st.status("Running Automated Pipeline...", expanded=True) as status:
            try:
                # ---------------------------------------------------------
                # REAL PIPELINE EXECUTION
                # ---------------------------------------------------------
                from scripts.run_automated_pipeline import AutomatedPipeline, PipelineConfig
                
                st.write("Initializing Pipeline Config...")
                config = PipelineConfig(
                    wsl_ip=wsl_ip,
                    headless=headless,
                    output_dir="outputs"
                )
                pipeline = AutomatedPipeline(config)
                
                st.write("Generating Scenario & Injecting Faults (LLM)...")
                
                # Prepare normalized incident dict with fallback field mappings
                # Supports both FAA format (report_id, description) and 
                # legacy format (incident_id, summary)
                rec = st.session_state.all_records
                report_id = str(rec.get("report_id", rec.get("incident_id", "UI_INPUT")))
                description_text = str(rec.get("description", rec.get("summary", "")))
                
                incident = {
                    "report_id": report_id,
                    "date": str(rec.get("date", "")),
                    "city": str(rec.get("city", "")),
                    "state": str(rec.get("state", "")),
                    "description": description_text,
                    "summary": description_text,  # Provide both for compatibility
                    "incident_type": str(rec.get("incident_type", "uas_sighting")),
                }
                
                # Execute
                st.write("Executing PX4 Flight Mission (This may take 2-5 mins)...")
                paths = pipeline.run_from_incident(incident=incident, skip_px4=False)
                
                # Save the original input data for traceability
                if paths.get("report_dir"):
                    input_file = Path(paths["report_dir"]) / "input" / "original_input.json"
                    input_file.parent.mkdir(exist_ok=True)
                    with open(input_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "metadata": {
                                "source": "UI File Upload",
                                "uploaded_at": str(pd.Timestamp.now()),
                            },
                            "original_record": rec,
                            "normalized_incident": incident,
                        }, f, indent=2, default=str)
                    paths["input_file"] = input_file
                
                st.session_state.pipeline_results = paths
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.success("Pipeline finished successfully.")
                
            except Exception as e:
                status.update(label="Pipeline Failed", state="error", expanded=True)
                st.error(f"❌ Analysis Failed: {str(e)}")
                
                with st.expander("🔍 View Technical Details & Troubleshooting"):
                    st.info("Possible Causes:\n- WSL is not running or IP is incorrect.\n- PX4/Gazebo failed to launch.\n- OpenAI API key is missing or invalid.")
                    import traceback
                    st.code(traceback.format_exc())
            finally:
                st.session_state.processing = False

# Step 4: Downloads
if st.session_state.pipeline_results:
    st.markdown("### 4. Results & Downloads")
    
    paths = st.session_state.pipeline_results
    
    # Show output directory path for manual inspection
    report_dir = paths.get("report_dir")
    if report_dir:
        st.info(f"📁 **Output saved to:** `{report_dir}`")
        st.caption("All files are saved to disk. You can browse this folder to view input, output, reports, and evaluation files.")
    
    cols = st.columns(4)
    
    # 1. Generated Parameters (Config)
    with cols[0]:
        p = paths.get("full_config")
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(
                    "📥 Generated Params", 
                    f, 
                    file_name="simulation_config.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.button("No Config", disabled=True, use_container_width=True)

    # 2. Telemetry Captured
    with cols[1]:
        p = paths.get("full_telemetry")
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(
                    "📥 Telemetry Data", 
                    f, 
                    file_name="telemetry_log.json", 
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.button("No Telemetry", disabled=True, use_container_width=True)
            
    # 3. Safety Report (PDF)
    with cols[2]:
        p = paths.get("pdf")
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(
                    "📄 Safety Report (PDF)", 
                    f, 
                    file_name="safety_report.pdf", 
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            # Fallback to JSON report
            p_json = paths.get("json")
            if p_json and os.path.exists(p_json):
                with open(p_json, "rb") as f:
                    st.download_button(
                        "📄 Safety Report (JSON)", 
                        f, 
                        file_name="safety_report.json", 
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.button("No Report", disabled=True, use_container_width=True)

    # 4. Evaluation (Excel)
    with cols[3]:
        p = paths.get("evaluation_excel")
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                st.download_button(
                    "📊 Evaluation (XLSX)", 
                    f, 
                    file_name="evaluation.xlsx", 
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
             # Fallback to JSON eval
            p_eval_json = paths.get("evaluation_json")
            if p_eval_json and os.path.exists(p_eval_json):
                with open(p_eval_json, "rb") as f:
                    st.download_button(
                        "📊 Evaluation (JSON)", 
                        f, 
                        file_name="evaluation.json", 
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.button("No Evaluation", disabled=True, use_container_width=True)

    # Show file structure for reference
    with st.expander("📂 View Output File Structure"):
        report_dir = paths.get("report_dir")
        if report_dir:
            dir_name = os.path.basename(report_dir)
            st.code(f"""outputs/{dir_name}/
├── input/
│   └── original_input.json                       (Original uploaded data)
├── generated/
│   ├── full_configuration_output_from_llm.json   (LLM-generated parameters)
│   └── full_telemetry_of_each_flight.json        (Flight telemetry data)
├── report/
│   ├── report.json                               (Safety report - JSON)
│   └── report.pdf                                (Safety report - PDF)
└── evaluation/
    ├── evaluation.json                           (ESRI metrics - JSON)
    └── evaluation_{dir_name}.xlsx                (ESRI metrics - Excel)""", language="text")
            
            st.markdown("**Quick Navigation:**")
            st.caption(f"Open folder: `{report_dir}`")

# Footer
st.markdown("---")
st.markdown("<center style='color:#64748B; font-size:12px'>AeroGuardian © 2026</center>", unsafe_allow_html=True)
