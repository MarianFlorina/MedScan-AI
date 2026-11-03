"""
UI theme for MedScan-AI (CSS + small HTML helpers).

Kept separate from app.py so the stylesheet is easy to tweak and the app file
stays focused on behaviour.  All dynamic text that reaches unsafe_allow_html
must go through html.escape (see app.py).
"""

THEME_CSS = """
<style>
:root {
  --bg: #0a0e17;
  --surface: #111827;
  --surface-2: #0d1320;
  --border: #1f2937;
  --border-strong: #2b3648;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --faint: #6b7280;
  --accent: #38bdf8;
  --accent-2: #6366f1;
  --ok: #34d399;
  --warn: #fbbf24;
  --danger: #f87171;
  --info: #60a5fa;
  --radius: 14px;
}

/* ── Base ── */
html, body, .stApp, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Helvetica Neue", Arial, sans-serif;
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #1b2440 0%, transparent 60%), var(--bg); color: var(--text); }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #0c121d; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #f3f4f6 !important; letter-spacing: -0.01em; }
h1 { font-size: 1.7rem !important; font-weight: 750 !important; }
h2 { font-size: 1.2rem !important; font-weight: 650 !important; }
h3 { font-size: 1.02rem !important; font-weight: 650 !important; }
h4 { font-size: 0.95rem !important; font-weight: 600 !important; }
p, li, label, .stMarkdown { color: var(--text); }
a { color: var(--accent) !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
  background: var(--surface-2) !important; border: 1px solid var(--border-strong) !important;
  border-radius: 10px !important; color: var(--text) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(56,189,248,.18) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stRadio label p { color: var(--muted) !important; font-size: .82rem !important; font-weight: 550; }

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--border-strong);
  background: var(--surface); color: var(--text); transition: all .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--accent); color: #fff; transform: translateY(-1px); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%); border: none; color: #fff;
  box-shadow: 0 6px 18px rgba(99,102,241,.28);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
  filter: brightness(1.08); box-shadow: 0 8px 22px rgba(99,102,241,.38);
}
.stDownloadButton > button { background: rgba(56,189,248,.12) !important; border: 1px solid rgba(56,189,248,.35) !important; color: #bae6fd !important; }

/* ── Cards / containers ── */
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .login-card-marker) {
  background: linear-gradient(180deg, rgba(17,24,39,.92), rgba(13,19,32,.92));
  border: 1px solid var(--border-strong); padding: 10px 12px;
  box-shadow: 0 24px 60px rgba(0,0,0,.45);
}
div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 12px; }
div[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 600 !important; }
div[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: .76rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .04em; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f3f4f6 !important; font-size: 1.7rem !important; font-weight: 750 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); padding: 10px 18px; font-weight: 600; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important; height: 3px !important; }

/* ── Sidebar nav (radio styled as menu) ── */
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
  background: transparent; border: 1px solid transparent; border-radius: 10px;
  padding: 9px 12px; width: 100%; transition: all .12s ease; cursor: pointer;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(56,189,248,.08); }
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: linear-gradient(135deg, rgba(14,165,233,.18), rgba(99,102,241,.18)); border-color: rgba(99,102,241,.45);
}
[data-testid="stSidebar"] .stRadio label > div:first-child { display: none; }   /* hide radio dot */
[data-testid="stSidebar"] .stRadio label p { color: var(--text) !important; font-size: .92rem !important; font-weight: 600; margin: 0; }

/* ── Brand ── */
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
  width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; font-size: 1.25rem;
  background: linear-gradient(135deg, #0ea5e9, #6366f1); box-shadow: 0 8px 20px rgba(99,102,241,.35);
}
.brand-name { font-weight: 800; font-size: 1.15rem; color: #f9fafb; line-height: 1.1; }
.brand-sub { color: var(--muted); font-size: .74rem; }

/* ── Login hero ── */
.login-hero { padding: 8px 8px 8px 0; }
.login-hero h2 { font-size: 2.15rem !important; line-height: 1.15; font-weight: 800 !important; margin: 26px 0 12px 0; }
.login-hero h2 span { background: linear-gradient(90deg, var(--accent), #a78bfa); -webkit-background-clip: text; background-clip: text; color: transparent; }
.login-lead { color: var(--muted); font-size: 1rem; line-height: 1.6; max-width: 520px; margin-bottom: 26px; }
.feature-row { display: flex; gap: 14px; align-items: flex-start; margin: 14px 0; }
.feature-ico { width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center; flex-shrink: 0;
  background: rgba(56,189,248,.1); border: 1px solid rgba(56,189,248,.25); font-size: 1.05rem; }
.feature-title { font-weight: 650; color: #f3f4f6; font-size: .94rem; }
.feature-desc { color: var(--muted); font-size: .84rem; line-height: 1.45; }
.login-foot { margin-top: 28px; color: var(--faint); font-size: .76rem; }
.login-title { font-size: 1.25rem; font-weight: 750; color: #f9fafb; margin: 4px 0 2px 0; }
.login-sub { color: var(--muted); font-size: .85rem; margin-bottom: 8px; }
.notice { border-radius: 10px; padding: 10px 12px; font-size: .82rem; margin-top: 10px; }
.notice-info { background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.28); color: #bae6fd; }
.notice-warn { background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.3); color: #fde68a; }

/* ── Home ── */
.hero {
  position: relative; overflow: hidden; border-radius: 20px; padding: 28px 30px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(14,165,233,.16), rgba(99,102,241,.2)), var(--surface);
  border: 1px solid var(--border-strong);
}
.hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 240px; height: 240px; border-radius: 50%;
  background: radial-gradient(circle, rgba(99,102,241,.35), transparent 70%); }
.hero h1 { margin: 0 0 6px 0 !important; font-size: 1.8rem !important; }
.hero p { color: #cbd5e1; margin: 0; max-width: 640px; font-size: .96rem; line-height: 1.55; position: relative; z-index: 1; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px; font-size: .74rem; font-weight: 650;
  border: 1px solid var(--border-strong); background: rgba(255,255,255,.04); color: var(--text); margin-right: 6px; }
.pill-ok { color: var(--ok); border-color: rgba(52,211,153,.4); background: rgba(52,211,153,.08); }
.pill-warn { color: var(--warn); border-color: rgba(251,191,36,.4); background: rgba(251,191,36,.08); }
.pill-bad { color: var(--danger); border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.08); }
.section-title { font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 22px 0 10px 2px; }
.step-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 14px 12px; height: 100%; }
.step-no { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-size: .78rem; font-weight: 800;
  background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff; margin-bottom: 8px; }
.step-title { font-weight: 650; font-size: .9rem; color: #f3f4f6; }
.step-desc { color: var(--muted); font-size: .79rem; line-height: 1.4; margin-top: 3px; }
.status-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid var(--border); font-size: .86rem; }
.status-row:last-child { border-bottom: none; }
.status-key { color: var(--muted); }
.recent-item { display: flex; justify-content: space-between; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: .85rem; }
.recent-item:last-child { border-bottom: none; }
.empty-state { text-align: center; padding: 34px 20px; color: var(--muted); }
.empty-state-icon { font-size: 2.2rem; margin-bottom: 8px; opacity: .6; }
.empty-state-title { color: #f3f4f6; font-size: 1.02rem; font-weight: 650; margin-bottom: 4px; }
.lock-card { text-align: center; padding: 40px 20px; border: 1px dashed var(--border-strong); border-radius: var(--radius); background: var(--surface); }

/* ── Sidebar bits ── */
.sidebar-user { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; margin: 14px 0 12px 0; }
.sidebar-label { color: var(--faint); font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin: 14px 0 6px 2px; }
.sidebar-foot { color: var(--faint); font-size: .7rem; line-height: 1.6; margin-top: 18px; }

/* ── Results ── */
.hipaa-banner { background: rgba(251,191,36,.07); border: 1px solid rgba(251,191,36,.28); border-radius: 10px;
  padding: 9px 14px; font-size: .8rem; color: #fde68a; margin-bottom: 16px; }
.entity-table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
.entity-table th { color: var(--muted); font-size: .72rem; font-weight: 700; text-align: left; padding: 8px 12px;
  border-bottom: 1px solid var(--border-strong); text-transform: uppercase; letter-spacing: .06em; }
.entity-table td { color: var(--text); font-size: .86rem; padding: 9px 12px; border-bottom: 1px solid var(--border); }
.entity-table tr:hover td { background: rgba(255,255,255,.03); }
.badge { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .71rem; font-weight: 700; letter-spacing: .03em; }
.badge-drug { background: rgba(56,189,248,.12); color: #7dd3fc; border: 1px solid rgba(56,189,248,.35); }
.badge-dosage { background: rgba(52,211,153,.12); color: #6ee7b7; border: 1px solid rgba(52,211,153,.35); }
.badge-condition { background: rgba(251,191,36,.12); color: #fcd34d; border: 1px solid rgba(251,191,36,.35); }
.badge-procedure { background: rgba(167,139,250,.12); color: #c4b5fd; border: 1px solid rgba(167,139,250,.35); }
.badge-history { background: rgba(156,163,175,.12); color: #d1d5db; border: 1px solid rgba(156,163,175,.3); }
.badge-frequency { background: rgba(244,114,182,.12); color: #f9a8d4; border: 1px solid rgba(244,114,182,.35); }
.badge-duration { background: rgba(251,146,60,.12); color: #fdba74; border: 1px solid rgba(251,146,60,.35); }
.badge-route { background: rgba(45,212,191,.12); color: #5eead4; border: 1px solid rgba(45,212,191,.35); }
.severity-card { border-radius: 12px; padding: 13px 16px; margin: 8px 0; border: 1px solid; font-size: .88rem; }
.severity-critical { background: rgba(248,113,113,.09); border-color: rgba(248,113,113,.4); border-left: 4px solid var(--danger); }
.severity-high { background: rgba(251,191,36,.08); border-color: rgba(251,191,36,.35); border-left: 4px solid var(--warn); }
.severity-moderate { background: rgba(96,165,250,.08); border-color: rgba(96,165,250,.35); border-left: 4px solid var(--info); }
.severity-low { background: rgba(52,211,153,.07); border-color: rgba(52,211,153,.3); border-left: 4px solid var(--ok); }
.severity-label { font-weight: 700; font-size: .83rem; color: #f3f4f6; }
.severity-desc { color: #d1d5db; margin-top: 4px; }
.severity-rec { color: var(--muted); font-size: .82rem; margin-top: 4px; }
.rel-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 11px 15px; margin: 6px 0; display: flex; align-items: center; gap: 12px; }
.rel-drug { color: #7dd3fc; font-weight: 650; } .rel-dosage { color: #6ee7b7; } .rel-arrow { color: var(--faint); }
.rel-condition { color: #fcd34d; font-weight: 550; } .rel-type { color: var(--muted); font-size: .78rem; }

/* ── Mobile ── */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .login-hero h2 { font-size: 1.6rem !important; }
  .hero { padding: 20px; } .hero h1 { font-size: 1.4rem !important; }
  .entity-table td, .entity-table th { padding: 7px 8px; font-size: .8rem; }
}
</style>
"""


def feature_row(icon: str, title: str, desc: str) -> str:
    return (
        f'<div class="feature-row"><div class="feature-ico">{icon}</div>'
        f'<div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div></div>'
    )


def step_card(no: int, title: str, desc: str) -> str:
    return (
        f'<div class="step-card"><div class="step-no">{no}</div>'
        f'<div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>'
    )
