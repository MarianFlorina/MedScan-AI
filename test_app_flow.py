"""
End-to-end UI logic tests (headless, no browser)
=================================================
Drives the real Streamlit app with streamlit.testing.AppTest against a TEMPORARY
user database and audit directory (your real users.json / audit_logs are untouched).

Checks: registration & roles, sign-in errors, permission enforcement (viewer vs
analyst vs admin), result persistence across reruns, safety alerts in the UI,
the AI assistant, HTML escaping, inactivity timeout and sign-out.

Run:  python test_app_flow.py
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

_TMP = tempfile.mkdtemp(prefix="medscan_test_")
os.environ["MEDSCAN_USERS_FILE"] = os.path.join(_TMP, "users.json")
os.environ["MEDSCAN_AUDIT_DIR"] = os.path.join(_TMP, "audit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from module7_hipaa import UserDatabase  # noqa: E402

PASS, FAIL = 0, 0
PW = "Str0ng!Passw0rd"


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def new_app():
    return AppTest.from_file("app.py", default_timeout=240).run()


def page_text(at):
    """All visible markdown / text / info in the main area (HTML included)."""
    parts = [m.value for m in at.markdown] + [t.value for t in at.text] + [e.value for e in at.error] \
        + [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def sign_in(at, who):
    at.text_input[0].set_value(who).run()
    at.text_input[1].set_value(PW)
    [b for b in at.button if b.label == "Sign in"][0].click().run()
    return at


def goto(at, page_label_part):
    radio = at.sidebar.radio(key="nav")
    target = [o for o in radio.options if page_label_part in o][0]
    radio.set_value(target).run()
    return at


# ---------------------------------------------------------------
print("\n1. Login page")
at = new_app()
check("renders without exception", not at.exception, [e.value for e in at.exception])
check("hero text present", "MedScan-AI" in page_text(at))
check("has Sign in + Create account tabs", [t.label for t in at.tabs] == ["Sign in", "Create account"])
check("no auto-login via URL password parameters", "pass" not in open("app.py", encoding="utf-8").read().lower().split("query_params")[-1][:400]
      if "query_params" in open("app.py", encoding="utf-8").read() else True)

print("\n2. Sign-in validation")
[b for b in at.button if b.label == "Sign in"][0].click().run()
check("empty form -> error", any("enter your username" in e.value.lower() for e in at.error))
at.text_input[0].set_value("nobody").run()
at.text_input[1].set_value("wrong")
[b for b in at.button if b.label == "Sign in"][0].click().run()
check("unknown user -> error", any("invalid" in e.value.lower() for e in at.error))

print("\n3. Registration (first account becomes admin)")
at = new_app()
at.text_input(key="reg_name").set_value("Alice Admin")
at.text_input(key="reg_email").set_value("alice@hospital.org")
at.text_input(key="reg_user").set_value("alice")
at.text_input(key="reg_pass").set_value(PW)
at.selectbox(key="reg_role").set_value("viewer")
at.button(key="reg_btn").click().run()
check("registration succeeds", any("created" in s.value.lower() for s in at.success), [e.value for e in at.error])
db = UserDatabase()
check("first user promoted to admin", db.get_user("alice")["role"] == "admin")
ok1, _ = db.register_user("vera", "vera@hospital.org", PW, "Vera Viewer", "viewer")
ok2, _ = db.register_user("andy", "andy@hospital.org", PW, "Andy Analyst", "analyst")
check("viewer + analyst accounts created", ok1 and ok2)
ok3, msg3 = db.register_user("dup", "alice@hospital.org", PW, "Dup", "viewer")
check("duplicate email rejected", not ok3, msg3)
ok4, _ = db.register_user("weak", "weak@hospital.org", "password", "Weak", "viewer")
check("weak password rejected", not ok4)

print("\n4. Viewer permissions")
at = sign_in(new_app(), "vera")
check("viewer signed in -> Home", any("Welcome back, Vera" in m.value for m in at.markdown), page_text(at)[:200])
goto(at, "Upload")
check("viewer sees lock message", "Uploading needs the Analyst or Admin role" in page_text(at))
check("viewer has NO file uploader", len(at.get("file_uploader")) == 0)
goto(at, "Demo")
at.button[[b.label for b in at.button].index("🚀 Run analysis")].click() if False else None
run_btn = [b for b in at.button if "Run analysis" in b.label][0]
run_btn.click().run()
check("viewer can run the demo", "Anomaly" in page_text(at) or "Safety" in page_text(at) or len(at.metric) >= 3)

print("\n5. Demo: safety alert + persistence across reruns")
at = sign_in(new_app(), "andy")
goto(at, "Demo")
at.selectbox[0].set_value("Drug interaction").run()
check("scenario text updates", "Warfarin" in at.text_area[0].value, at.text_area[0].value[:60])
[b for b in at.button if "Run analysis" in b.label][0].click().run()
txt = page_text(at)
check("no exception after analysis", not at.exception, [e.value for e in at.exception])
check("CRITICAL interaction alert is shown", "CRITICAL" in txt.upper())
check("entities table shows Warfarin", "Warfarin" in txt)
at.run()  # an unrelated rerun must not wipe the results
check("results survive a rerun", "Warfarin" in page_text(at) and "CRITICAL" in page_text(at).upper())
goto(at, "Home")
check("home counts the analysis", any(m.label == "Analyses run" and m.value == "1" for m in at.metric))

print("\n6. Analyst/Admin permissions")
goto(at, "Upload")
check("analyst has file uploader", len(at.get("file_uploader")) == 1)
check("admin panels hidden from analyst", not any("Audit log" in e.label for e in at.sidebar.expander))
at_admin = sign_in(new_app(), "alice")
check("admin sees audit + users panels", {"📊 Audit log", "👥 Users"} <= {e.label for e in at_admin.sidebar.expander})

print("6b. Admin: create admin, change roles")
db = UserDatabase()
ok, _ = db.register_user("bob", "bob@hospital.org", PW, "Bob Boss", "admin")
check("admin-role user can be created", ok and db.get_user("bob")["role"] == "admin")
okr, msg = db.set_role("andy", "admin")
check("set_role promotes analyst to admin", okr and db.get_user("andy")["role"] == "admin", msg)
okr, _ = db.set_role("andy", "analyst")
check("set_role demotes admin back", okr and db.get_user("andy")["role"] == "analyst")
db.set_role("bob", "viewer")
okr, msg = db.set_role("alice", "viewer")
check("last admin cannot be demoted", not okr and "last admin" in msg.lower(), msg)
check("invalid role rejected", not db.set_role("vera", "root")[0])
at_b = sign_in(new_app(), "alice")
check("admin UI has Add user + Change role controls",
      any(b.label == "Update role" for b in at_b.sidebar.button) and any(sb.key == "admin_new_role" for sb in at_b.sidebar.selectbox))
check("self-registration still offers only viewer/analyst", (lambda o: len(o) == 2 and not any("admin" in x.lower() for x in o))(list(new_app().selectbox(key="reg_role").options)))

print("\n7. AI assistant uses the analysis")
goto(at, "AI assistant")
at.text_input(key="ai_q_input").set_value("What medications are prescribed?")
at.button(key="ai_ask_btn").click().run()
check("answer mentions the prescribed drugs", "arfarin" in page_text(at), page_text(at)[-300:])
check("answer is rendered as markdown (no raw ** in HTML div)", "<div" not in "".join(m.value for m in at.markdown if "Q:" in m.value))

print("\n8. HTML escaping (XSS)")
goto(at, "Demo")
at.text_area[0].set_value("Patient prescribed Warfarin 5mg <script>alert(1)</script>").run()
[b for b in at.button if "Run analysis" in b.label][0].click().run()
check("no raw <script> tag in any unsafe-HTML block",
      not any("<script>alert" in m.value for m in at.markdown if "badge" in m.value or "entity-table" in m.value))

print("\n9. Inactivity timeout and sign-out")
at.session_state["last_activity"] = (datetime.utcnow() - timedelta(minutes=45)).isoformat()
at.run()
check("idle session is signed out", not at.session_state["authenticated"])
check("flash message explains why", any("inactivity" in w.value for w in at.warning))
at = sign_in(new_app(), "andy")
[b for b in at.sidebar.button if b.label == "Sign out"][0].click().run()
check("sign out returns to login", [t.label for t in at.tabs] == ["Sign in", "Create account"])

print("\n10. Audit trail")
events = UserDatabase and __import__("module7_hipaa").AuditLogger().get_logs(limit=1000)
kinds = {e["event_type"] for e in events}
check("logins and processing were logged", {"ACCESS", "PROCESSING"} <= kinds, kinds)

print(f"\n{'=' * 60}\n  RESULT: {PASS} passed, {FAIL} failed\n{'=' * 60}")
sys.exit(0 if FAIL == 0 else 1)
