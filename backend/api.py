"""
Poly — AI Flaky Test Trust Layer
FastAPI Backend Server
"""

from fastapi import FastAPI, Depends, HTTPException, Header, Query, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional
import os
import hashlib
import secrets
import time

from engine.trust_engine import (
    process_test_run, get_dashboard_data, get_test_detail,
    get_quarantined_tests, send_alert, _init_db, _get_db, _ensure_repo,
)

app = FastAPI(
    title="Poly — AI Flaky Test Trust Layer",
    description="Production-ready flaky test detection with Bayesian scoring, z-score anomaly detection, and environment-aware analysis",
    version="2.0.0",
)

API_KEY = os.environ.get("POLY_API_KEY", "poly-dev-key")
ADMIN_PASSWORD = os.environ.get("POLY_ADMIN_PASSWORD", "admin123")

DB_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DB_DIR, exist_ok=True)

base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_api_key(x_poly_api_key: Optional[str] = Header(None)):
    if x_poly_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_poly_api_key


def _get_admin_session(request: Request):
    token = request.cookies.get("poly_admin_token")
    if not token:
        return None
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT username, role FROM admin_users WHERE id=?", (int(token),))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def require_admin(request: Request):
    admin = _get_admin_session(request)
    if not admin:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return admin


class JUnitUpload(BaseModel):
    xml_content: str
    repo_name: str
    branch: str = "main"
    commit_sha: Optional[str] = None
    environment: Optional[str] = None


class RunInput(BaseModel):
    test_results: list[dict]
    repo_name: str
    run_id: Optional[str] = None
    branch: str = "main"
    commit_sha: Optional[str] = None
    environment: Optional[str] = None


class AlertConfig(BaseModel):
    webhook_url: str
    channel_type: str = "discord"
    min_trust_drop: float = 20
    alert_on_flaky: bool = True
    alert_on_quarantine: bool = True


class AdminLogin(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    name: str
    email: str
    github_username: Optional[str] = None
    plan: str = "free"
    referrer: Optional[str] = None
    signup_source: Optional[str] = None
    notes: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    github_username: Optional[str] = None
    plan: Optional[str] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


@app.on_event("startup")
def startup():
    _init_db()


@app.get("/", response_class=HTMLResponse)
def root():
    landing_path = os.path.join(base_path, "landing", "index.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path)
    return HTMLResponse("<h1>Poly — AI Flaky Test Trust Layer</h1><p>Landing page not found. Visit <a href='/dashboard/'>Dashboard</a> or <a href='/docs'>API Docs</a></p>", status_code=404)


# ===================== ADMIN AUTH =====================

@app.post("/api/admin/login")
def admin_login(data: AdminLogin, response: Response):
    pw_hash = hashlib.sha256(data.password.encode()).hexdigest()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id, username, role FROM admin_users WHERE username=? AND password_hash=?", (data.username, pw_hash))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    response.set_cookie(key="poly_admin_token", value=str(row["id"]), httponly=True, max_age=86400 * 7)
    return {"status": "ok", "username": row["username"], "role": row["role"]}


@app.post("/api/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie("poly_admin_token")
    return {"status": "ok"}


@app.get("/api/admin/me")
def admin_me(request: Request):
    admin = _get_admin_session(request)
    if not admin:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return admin


# ===================== ADMIN USERS API =====================

@app.get("/api/admin/users")
def admin_list_users(request: Request, search: str = "", plan: str = "", sort: str = "newest", page: int = 1, per_page: int = 20):
    require_admin(request)
    conn = _get_db()
    c = conn.cursor()
    where = []
    params = []
    if search:
        where.append("(name LIKE ? OR email LIKE ? OR github_username LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    if plan:
        where.append("plan=?")
        params.append(plan)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    order = "DESC" if sort == "newest" else "ASC"
    offset = (page - 1) * per_page

    c.execute(f"SELECT COUNT(*) as cnt FROM users {where_sql}", params)
    total = c.fetchone()["cnt"]

    c.execute(
        f"SELECT * FROM users {where_sql} ORDER BY created_at {order} LIMIT ? OFFSET ?",
        params + [per_page, offset]
    )
    users = [dict(r) for r in c.fetchall()]

    # Signup source breakdown
    c.execute("SELECT signup_source, COUNT(*) as cnt FROM users GROUP BY signup_source ORDER BY cnt DESC")
    sources = [dict(r) for r in c.fetchall()]

    # Plan breakdown
    c.execute("SELECT plan, COUNT(*) as cnt FROM users GROUP BY plan ORDER BY cnt DESC")
    plans = [dict(r) for r in c.fetchall()]

    # Daily signups (last 30 days)
    c.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as cnt FROM users
        WHERE created_at >= datetime('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY day
    """)
    daily = [dict(r) for r in c.fetchall()]

    # Recent activity
    c.execute("""
        SELECT ua.action, ua.detail, ua.created_at, u.name, u.email
        FROM user_activity ua LEFT JOIN users u ON ua.user_id = u.id
        ORDER BY ua.created_at DESC LIMIT 10
    """)
    activity = [dict(r) for r in c.fetchall()]

    conn.close()
    return {
        "users": users, "total": total, "page": page, "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "sources": sources, "plans": plans, "daily_signups": daily,
        "recent_activity": activity,
    }


@app.post("/api/admin/users")
def admin_create_user(data: UserCreate, request: Request):
    require_admin(request)
    api_key = f"poly_{secrets.token_urlsafe(24)}"
    conn = _get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (name, email, github_username, api_key, plan, referrer, signup_source, notes) VALUES (?,?,?,?,?,?,?,?)",
            (data.name, data.email, data.github_username, api_key, data.plan, data.referrer, data.signup_source, data.notes)
        )
        user_id = c.lastrowid
        c.execute("INSERT INTO user_activity (user_id, action, detail) VALUES (?,?,?)",
                   (user_id, "signup", f"Created by admin | source: {data.signup_source or 'manual'}"))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))
    conn.close()
    return {"status": "ok", "user_id": user_id, "api_key": api_key}


@app.put("/api/admin/users/{user_id}")
def admin_update_user(user_id: int, data: UserUpdate, request: Request):
    require_admin(request)
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE id=?", (user_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    updates = []
    params = []
    for field, value in data.model_dump().items():
        if value is not None:
            updates.append(f"{field}=?")
            params.append(value)
    if not updates:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")
    params.append(user_id)
    c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", params)
    c.execute("INSERT INTO user_activity (user_id, action, detail) VALUES (?,?,?)",
               (user_id, "updated", f"Updated by admin: {', '.join(updates)}"))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, request: Request):
    require_admin(request)
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT name, email FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    c.execute("DELETE FROM user_activity WHERE user_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted": row["name"]}


@app.get("/api/admin/stats")
def admin_stats(request: Request):
    require_admin(request)
    conn = _get_db()
    c = conn.cursor()
    stats = {}
    c.execute("SELECT COUNT(*) as cnt FROM users")
    stats["total_users"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE plan='pro'")
    stats["pro_users"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE plan='enterprise'")
    stats["enterprise_users"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE is_active=1")
    stats["active_users"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE created_at >= datetime('now', '-7 days')")
    stats["new_this_week"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM users WHERE created_at >= datetime('now', '-1 day')")
    stats["new_today"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(DISTINCT repo_id) as cnt FROM ci_runs")
    stats["total_repos"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM ci_runs WHERE timestamp >= datetime('now', '-24 hours')")
    stats["runs_today"] = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM test_results")
    stats["total_test_results"] = c.fetchone()["cnt"]

    # Top referrers
    c.execute("SELECT referrer, COUNT(*) as cnt FROM users WHERE referrer IS NOT NULL GROUP BY referrer ORDER BY cnt DESC LIMIT 5")
    stats["top_referrers"] = [dict(r) for r in c.fetchall()]

    # Top signup sources
    c.execute("SELECT signup_source, COUNT(*) as cnt FROM users WHERE signup_source IS NOT NULL GROUP BY signup_source ORDER BY cnt DESC LIMIT 5")
    stats["top_sources"] = [dict(r) for r in c.fetchall()]

    conn.close()
    return stats


# ===================== EXISTING API ROUTES =====================

@app.post("/api/junit", dependencies=[Depends(verify_api_key)])
def ingest_junit(data: JUnitUpload):
    result = process_test_run(
        xml_content=data.xml_content,
        repo_name=data.repo_name,
        branch=data.branch,
        commit_sha=data.commit_sha,
        environment=data.environment,
    )
    return result


@app.post("/api/runs", dependencies=[Depends(verify_api_key)])
def create_run(data: RunInput):
    import xml.etree.ElementTree as ET
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(testsuites, "testsuite", name="custom", tests=str(len(data.test_results)))
    for t in data.test_results:
        attrs = {"name": t.get("name", "unknown"), "classname": t.get("classname", ""), "time": str(t.get("duration", 0))}
        tc = ET.SubElement(testsuite, "testcase", **attrs)
        if t.get("status") == "failed":
            ET.SubElement(tc, "failure", message=t.get("error_message", "Test failed"))
        elif t.get("status") == "skipped":
            ET.SubElement(tc, "skipped")
    xml_str = ET.tostring(testsuites, encoding="unicode")
    result = process_test_run(xml_content=xml_str, repo_name=data.repo_name, run_id=data.run_id, branch=data.branch, commit_sha=data.commit_sha, environment=data.environment)
    return result


@app.get("/api/dashboard")
def dashboard(repo_name: str = Query(...)):
    return get_dashboard_data(repo_name)


@app.get("/api/tests")
def list_tests(repo_name: str = Query(...)):
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM repositories WHERE name=?", (repo_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    repo_id = row["id"]
    c.execute(
        "SELECT test_name, AVG(trust_score) as trust_score, COUNT(*) as runs, flaky_category, "
        "AVG(CASE WHEN status='passed' THEN 1.0 ELSE 0.0 END) as pass_rate, AVG(duration) as avg_duration "
        "FROM test_results WHERE repo_id=? GROUP BY test_name ORDER BY trust_score ASC",
        (repo_id,)
    )
    tests = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"repo": repo_name, "tests": tests, "total": len(tests)}


@app.get("/api/tests/{test_name:path}")
def test_detail(test_name: str, repo_name: str = Query(...)):
    result = get_test_detail(repo_name, test_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/quarantined")
def quarantined(repo_name: str = Query(...), threshold: float = Query(30)):
    return {"repo": repo_name, "threshold": threshold, "quarantined": get_quarantined_tests(repo_name, threshold)}


@app.get("/api/runs")
def list_runs(repo_name: str = Query(...), limit: int = Query(20, le=100)):
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM repositories WHERE name=?", (repo_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    repo_id = row["id"]
    c.execute(
        "SELECT run_id, branch, commit_sha, total_tests, passed, failed, avg_trust_score, timestamp "
        "FROM ci_runs WHERE repo_id=? ORDER BY timestamp DESC LIMIT ?",
        (repo_id, limit)
    )
    runs = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"repo": repo_name, "runs": runs}


@app.get("/api/repos")
def list_repos():
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute(
        "SELECT r.name, COUNT(DISTINCT tr.run_id) as total_runs, COUNT(DISTINCT tr.test_name) as total_tests, AVG(tr.trust_score) as avg_trust "
        "FROM repositories r LEFT JOIN test_results tr ON r.id = tr.repo_id GROUP BY r.name ORDER BY r.name"
    )
    repos = [dict(r) for r in c.fetchall()]
    conn.close()
    return {"repos": repos}


@app.post("/api/alerts/config", dependencies=[Depends(verify_api_key)])
def set_alert_config(repo_name: str = Query(...), config: AlertConfig = ...):
    _init_db()
    conn = _get_db()
    repo_id = _ensure_repo(conn, repo_name)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO alerts_config (repo_id, webhook_url, channel_type, min_trust_drop, alert_on_flaky, alert_on_quarantine) VALUES (?,?,?,?,?,?)",
        (repo_id, config.webhook_url, config.channel_type, config.min_trust_drop, config.alert_on_flaky, config.alert_on_quarantine)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "repo": repo_name}


@app.post("/api/alerts/test", dependencies=[Depends(verify_api_key)])
def test_alert(repo_name: str = Query(...)):
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM alerts_config WHERE repo_id=(SELECT id FROM repositories WHERE name=?)", (repo_name,))
    cfg = c.fetchone()
    conn.close()
    if not cfg:
        raise HTTPException(status_code=404, detail="No alert config found")
    ok = send_alert(repo_name=repo_name, webhook_url=cfg["webhook_url"], channel_type=cfg["channel_type"], alert_data={"Test": "Poly connectivity test", "Status": "Alert channel working"})
    return {"status": "sent" if ok else "failed"}


@app.delete("/api/tests/{test_name:path}", dependencies=[Depends(verify_api_key)])
def delete_test(test_name: str, repo_name: str = Query(...)):
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM repositories WHERE name=?", (repo_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    c.execute("DELETE FROM test_results WHERE repo_id=? AND test_name=?", (row["id"], test_name))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted": deleted}


@app.get("/badge/{repo_name}")
def trust_badge(repo_name: str):
    _init_db()
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM repositories WHERE name=?", (repo_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Repository not found")
    c.execute("SELECT AVG(trust_score) as avg_trust FROM test_results WHERE repo_id=?", (row["id"],))
    score = c.fetchone()["avg_trust"] or 100
    conn.close()
    score = round(score, 0)
    if score >= 90: color = "#22c55e"
    elif score >= 70: color = "#eab308"
    elif score >= 50: color = "#f97316"
    else: color = "#ef4444"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="220" height="30"><rect width="220" height="30" rx="6" fill="#0f0f11"/><rect x="110" width="110" height="30" rx="6" fill="{color}"/><text x="10" y="20" fill="#a1a1aa" font-family="system-ui,sans-serif" font-size="11" font-weight="600">poly trust</text><text x="120" y="20" fill="#fff" font-family="system-ui,sans-serif" font-size="11" font-weight="700">{int(score)}%</text></svg>'
    return HTMLResponse(content=svg, media_type="image/svg+xml")


# ===================== PAGE ROUTES =====================

@app.get("/dashboard/", response_class=HTMLResponse)
def serve_dashboard():
    return FileResponse(os.path.join(base_path, "dashboard", "index.html"))


@app.get("/dashboard/test-detail.html", response_class=HTMLResponse)
def serve_test_detail():
    path = os.path.join(base_path, "dashboard", "test-detail.html")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse("<h1>Test Detail</h1>", status_code=404)


@app.get("/dashboard/guide.html", response_class=HTMLResponse)
def serve_guide():
    path = os.path.join(base_path, "dashboard", "guide.html")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse("<h1>Guide</h1>", status_code=404)


@app.get("/admin/", response_class=HTMLResponse)
def serve_admin():
    return FileResponse(os.path.join(base_path, "dashboard", "admin.html"))


@app.get("/landing/", response_class=HTMLResponse)
def serve_landing():
    return RedirectResponse(url="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)