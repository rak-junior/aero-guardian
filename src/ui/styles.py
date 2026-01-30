"""
AeroGuardian UI Styles - Premium Design
=======================================
Author: AeroGuardian Member
Date: 2026-01-30

Modern, clean CSS for competition-ready Streamlit demo.
"""

# Modern color palette
COLORS = {
    "primary": "#6366F1",
    "primary_light": "#818CF8",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "bg_main": "#F1F5F9",
    "bg_card": "#FFFFFF",
    "text_primary": "#0F172A",
    "text_secondary": "#64748B",
    "border": "#E2E8F0",
}

# Premium CSS
CUSTOM_CSS = """
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Reset */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main app background */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 50%, #F1F5F9 100%);
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Container padding */
.block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px;
}

/* Card styling */
div[data-testid="stVerticalBlock"] > div {
    background: transparent;
}

/* File uploader */
.stFileUploader {
    background: white;
    border-radius: 12px;
    border: 2px dashed #E2E8F0;
    padding: 1rem;
}

.stFileUploader:hover {
    border-color: #6366F1;
    background: #F8FAFC;
}

.stFileUploader > div > div > div > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: white;
    color: #6366F1;
    border: 2px solid #6366F1;
    box-shadow: none;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.stDataFrame > div > div > div > div {
    font-size: 12px;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
}

/* Checkbox styling */
.stCheckbox {
    background: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366F1, #8B5CF6, #A78BFA);
    border-radius: 10px;
}

/* Expander */
.streamlit-expanderHeader {
    background: white;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

/* Info/Warning/Error boxes */
.stAlert {
    border-radius: 10px;
    border: none;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #F1F5F9;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366F1, #8B5CF6);
    border-radius: 3px;
}

/* Metric styling */
div[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Animation keyframes */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-pulse {
    animation: pulse 2s ease-in-out infinite;
}

.animate-slide {
    animation: slideIn 0.3s ease-out;
}
</style>
"""

def get_panel_html(title: str, icon: str, content: str) -> str:
    return f"""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    ">
        <div style="
            font-size: 12px;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        ">{icon} {title}</div>
        {content}
    </div>
    """

def get_log_html(entries: list) -> str:
    log_lines = []
    for entry in entries:
        level = entry.get("level", "info")
        msg = entry.get("message", "")
        time = entry.get("time", "")
        log_lines.append(f'<div class="log-{level}">[{time}] {msg}</div>')
    return f'<div class="log-container">{"".join(log_lines)}</div>'

def get_status_badge(status: str) -> str:
    colors = {
        "ready": ("#DCFCE7", "#166534"),
        "processing": ("#FEF3C7", "#92400E"),
        "complete": ("#DBEAFE", "#1E40AF"),
        "error": ("#FEE2E2", "#991B1B"),
    }
    bg, fg = colors.get(status, ("#F1F5F9", "#475569"))
    return f'<span style="background:{bg};color:{fg};padding:6px 14px;border-radius:20px;font-size:12px;font-weight:600;">{status.title()}</span>'
