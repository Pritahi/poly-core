"""
Falsky — AI Flaky Test Trust Layer
Core Trust Engine v2.0: Production-ready flaky detection with Bayesian scoring,
z-score anomaly detection, environment tracking, and sequential analysis.

Refactored to use Supabase SDK instead of psycopg2.
"""

import xml.etree.ElementTree as ET
import json
import hashlib
import time
import math
import os
import logging
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Optional

# Supabase SDK
from engine.db import get_client, table, rpc, select, select_one, insert, update, delete, upsert

logger = logging.getLogger("falsky.engine")


def _test_hash(test_name, repo_name):
    return hashlib.sha256(f"{repo_name}:{test_name}".encode()).hexdigest()[:16]


# ─────────────────────────────────────────────
# DB Init (skip if tables already exist)
# ─────────────────────────────────────────────

_db_initialized = False


def ensure_initialized():
    """Initialize DB — tables should already exist in Supabase."""
    global _db_initialized
    if _db_initialized:
        return
    try:
        # Test connection by querying repositories
        select("repositories", "id", limit=1)
        _db_initialized = True
        logger.info("Supabase connection verified")
    except Exception as e:
        logger.warning(f"DB not ready: {e}")


# ─────────────────────────────────────────────
# JUnit XML Parsing
# ─────────────────────────────────────────────

def parse_junit_xml(xml_content: str) -> dict:
    """Parse JUnit XML and extract test results."""
    results = {"tests": [], "total": 0, "passed": 0, "failed": 0, "skipped": 0, "duration": 0}
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return results

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "unknown")
            classname = testcase.get("classname", "")
            full_name = f"{classname}.{name}" if classname else name
            duration = float(testcase.get("time", 0))
            results["duration"] += duration
            results["total"] += 1

            status = "passed"
            error_message = ""
            error_type = ""

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None:
                status = "failed"
                error_message = (failure.get("message", "") or failure.text or "")[:500]
                error_type = "failure"
            elif error is not None:
                status = "failed"
                error_message = (error.get("message", "") or error.text or "")[:500]
                error_type = "error"
            elif skipped is not None:
                status = "skipped"

            if status == "passed":
                results["passed"] += 1
            elif status == "failed":
                results["failed"] += 1
            else:
                results["skipped"] += 1

            results["tests"].append({
                "name": full_name,
                "status": status,
                "duration": duration,
                "error_message": error_message,
                "error_type": error_type,
            })

    return results


def _get_test_history(repo_id, test_name, limit=50):
    """Get test history using Supabase SDK."""
    rows = select("test_results", 
                  "status, duration, error_message, run_timestamp, trust_score, environment",
                  filters={"repo_id": repo_id, "test_name": test_name},
                  order="-run_timestamp", limit=limit)
    return rows


# ─────────────────────────────────────────────
# Bayesian Pass Rate (Beta Distribution)
# ─────────────────────────────────────────────

def _beta_pdf(x, alpha, beta_param):
    if x <= 0 or x >= 1:
        return 0.0
    def ln_gamma(z):
        if z < 0.5:
            return math.log(math.pi / math.sin(math.pi * z)) - ln_gamma(1 - z)
        z -= 1
        g = 0.99999999999980993 + (
            676.5203681218851 / (z + 1) - 1259.1392167224028 / (z + 2) +
            771.32342877765313 / (z + 3) - 176.61502916214059 / (z + 4) +
            12.507343278686905 / (z + 5) - 0.13857109526572012 / (z + 6) +
            9.9843695780195716e-6 / (z + 7) + 1.5056327351493116e-7 / (z + 8)
        )
        return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(z + 7.5) - (z + 7.5) + math.log(g)
    log_beta = ln_gamma(alpha) + ln_gamma(beta_param) - ln_gamma(alpha + beta_param)
    log_pdf = (alpha - 1) * math.log(x) + (beta_param - 1) * math.log(1 - x) - log_beta
    return math.exp(log_pdf)


def calculate_bayesian_pass_rate(history, prior_alpha=2.0, prior_beta=2.0):
    if not history:
        return 0.5
    passed = sum(1 for h in history if h["status"] == "passed")
    failed = len(history) - passed
    alpha = prior_alpha + passed
    beta_p = prior_beta + failed
    if alpha + beta_p > 2:
        map_estimate = (alpha - 1) / (alpha + beta_p - 2)
    else:
        map_estimate = alpha / (alpha + beta_p)
    return max(0.0, min(1.0, map_estimate))


def calculate_score_confidence(history, prior_alpha=2.0, prior_beta=2.0):
    if not history:
        return 0.0
    passed = sum(1 for h in history if h["status"] == "passed")
    failed = len(history) - passed
    alpha = prior_alpha + passed
    beta_p = prior_beta + failed
    n = alpha + beta_p
    if n <= 0:
        return 0.0
    variance = (alpha * beta_p) / (n * n * (n + 1))
    std = math.sqrt(variance)
    confidence = max(0.0, min(1.0, 1.0 - std * 3.5))
    return round(confidence, 3)


def wilson_score_interval(history, confidence=0.95):
    if not history:
        return (0.25, 0.75, 0.5)
    n = len(history)
    p_hat = sum(1 for h in history if h["status"] == "passed") / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4), round(p_hat, 4))


def calculate_duration_variance(history):
    if len(history) < 3:
        return 0.0
    durations = [h["duration"] for h in history if h.get("duration") and h["duration"] > 0]
    if len(durations) < 3:
        return 0.0
    mean = sum(durations) / len(durations)
    variance = sum((d - mean) ** 2 for d in durations) / len(durations)
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else 0
    return min(cv * 100, 100)


def calculate_duration_zscores(history):
    durations = [h["duration"] for h in history if h.get("duration") and h["duration"] > 0]
    if len(durations) < 3:
        return []
    mean = sum(durations) / len(durations)
    variance = sum((d - mean) ** 2 for d in durations) / len(durations)
    std = math.sqrt(variance)
    if std == 0:
        return [(d, 0.0, False) for d in durations]
    results = []
    for d in durations:
        z = (d - mean) / std
        is_anomaly = abs(z) > 2.0
        results.append((d, round(z, 2), is_anomaly))
    return results


def duration_anomaly_rate(history):
    zscores = calculate_duration_zscores(history)
    if not zscores:
        return 0.0
    anomalies = sum(1 for _, _, is_anom in zscores if is_anom)
    return anomalies / len(zscores)


def calculate_recent_trend(history, window=5):
    if len(history) < window + 2:
        return 0.0
    recent = history[:window]
    older = history[window:2 * window]
    if not older:
        return 0.0
    recent_rate = calculate_bayesian_pass_rate(recent)
    older_rate = calculate_bayesian_pass_rate(older)
    return (recent_rate - older_rate) * 100


def calculate_error_pattern(history):
    if not history:
        return 0.0
    errors = [h["error_message"] for h in history if h["status"] == "failed" and h.get("error_message")]
    if len(errors) < 2:
        return 0.0
    def normalize(err):
        import re
        err = err.strip().lower()
        err = re.sub(r'line \d+', 'line N', err)
        err = re.sub(r'at .+:\d+', 'at FILE:LINE', err)
        err = re.sub(r'0x[0-9a-f]+', '0xADDR', err)
        return err
    normalized = [normalize(e) for e in errors]
    error_hashes = [hashlib.md5(e.encode()).hexdigest() for e in normalized]
    counter = Counter(error_hashes)
    most_common_ratio = counter.most_common(1)[0][1] / len(error_hashes)
    return most_common_ratio * 100


def get_unique_error_patterns(history, top_n=5):
    if not history:
        return []
    errors = [h["error_message"] for h in history if h["status"] == "failed" and h.get("error_message")]
    if not errors:
        return []
    def normalize(err):
        import re
        err = err.strip().lower()
        err = re.sub(r'line \d+', 'line N', err)
        err = re.sub(r'at .+:\d+', 'at FILE:LINE', err)
        err = re.sub(r'0x[0-9a-f]+', '0xADDR', err)
        return err
    pattern_counts = Counter()
    pattern_examples = {}
    for e in errors:
        norm = normalize(e)
        pattern_counts[norm] += 1
        if norm not in pattern_examples:
            pattern_examples[norm] = e[:200]
    return [
        {"pattern": pattern_examples[p], "normalized": p, "count": c, "percentage": round(c / len(errors) * 100, 1)}
        for p, c in pattern_counts.most_common(top_n)
    ]


def detect_sequential_pattern(history):
    if len(history) < 6:
        return None
    statuses = [1 if h["status"] == "passed" else 0 for h in history]
    n = len(statuses)
    transitions = sum(1 for i in range(n - 1) if statuses[i] != statuses[i + 1])
    transition_rate = transitions / (n - 1)
    if transition_rate > 0.7:
        return "alternating"
    runs = 1
    for i in range(1, n):
        if statuses[i] != statuses[i - 1]:
            runs += 1
    expected_runs = 1 + (2 * sum(statuses) * (n - sum(statuses))) / n
    if n > 1:
        runs_variance = (2 * sum(statuses) * (n - sum(statuses)) * (2 * sum(statuses) * (n - sum(statuses)) - n)) / (n * n * (n - 1))
        if runs_variance > 0:
            z = (runs - expected_runs) / math.sqrt(runs_variance)
            if z < -1.96:
                return "clustered"
    mean_s = sum(statuses) / n
    if mean_s == 0 or mean_s == 1:
        return None
    variance = sum((s - mean_s) ** 2 for s in statuses) / n
    if variance == 0:
        return None
    for lag in range(2, min(6, n // 2)):
        autocorr = sum((statuses[i] - mean_s) * (statuses[i + lag] - mean_s) for i in range(n - lag)) / (n * variance)
        if autocorr > 0.5:
            return f"periodic_{lag}"
    return None


def analyze_environment_correlation(history):
    env_groups = defaultdict(lambda: {"passed": 0, "total": 0, "durations": []})
    for h in history:
        env = h.get("environment") or "unknown"
        env_groups[env]["total"] += 1
        if h["status"] == "passed":
            env_groups[env]["passed"] += 1
        if h.get("duration"):
            env_groups[env]["durations"].append(h["duration"])
    if len(env_groups) < 2:
        return None
    rates = {}
    for env, data in env_groups.items():
        if data["total"] >= 2:
            rates[env] = {
                "pass_rate": data["passed"] / data["total"],
                "runs": data["total"],
                "avg_duration": sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0,
            }
    if len(rates) < 2:
        return None
    pass_rates = [r["pass_rate"] for r in rates.values()]
    max_diff = max(pass_rates) - min(pass_rates)
    if max_diff > 0.3:
        return rates
    return None


def classify_flaky_category(history, current_result=None):
    if not history or len(history) < 2:
        return None
    statuses = [h["status"] for h in history]
    pass_rate = sum(1 for s in statuses if s == "passed") / len(statuses)
    if pass_rate > 0.92 or pass_rate < 0.08:
        return None
    durations = [h["duration"] for h in history if h.get("duration") and h["duration"] > 0]
    errors = [h["error_message"] for h in history if h.get("error_message")]
    scores = {}
    if len(durations) >= 3:
        anomaly_rate = duration_anomaly_rate(history)
        cv = calculate_duration_variance(history) / 100
        timing_score = max(cv * 1.5, anomaly_rate * 2)
        if timing_score > 0.3:
            scores["timing"] = min(1.0, timing_score)
    pattern = detect_sequential_pattern(history)
    if pattern == "alternating":
        scores["order_dependency"] = 0.85
    elif pattern == "clustered":
        scores["shared_state"] = 0.7
    elif pattern and pattern.startswith("periodic_"):
        scores["order_dependency"] = 0.6
    if len(errors) >= 2:
        unique_errors = len(set(errors))
        error_ratio = unique_errors / len(errors)
        if error_ratio > 0.5:
            scores["non_deterministic_data"] = min(1.0, error_ratio)
    if len(errors) >= 2:
        error_consistency = calculate_error_pattern(history) / 100
        if error_consistency > 0.6 and 0.15 < pass_rate < 0.85:
            shared_score = error_consistency * (1 - abs(pass_rate - 0.5) * 0.5)
            if "shared_state" not in scores or shared_score > scores.get("shared_state", 0):
                scores["shared_state"] = shared_score
    env_corr = analyze_environment_correlation(history)
    if env_corr:
        scores["environment_specific"] = 0.8
    if not scores:
        return None
    best_category = max(scores, key=scores.get)
    return best_category


def calculate_trust_score(history):
    if not history:
        return 100.0
    bayesian_rate = calculate_bayesian_pass_rate(history)
    pass_score = bayesian_rate * 55
    trend = calculate_recent_trend(history)
    trend_score = max(0, (50 + trend)) / 100 * 20
    duration_var = calculate_duration_variance(history)
    anomaly_rate = duration_anomaly_rate(history)
    duration_penalty = min(duration_var, 100) * 0.07 + anomaly_rate * 30
    duration_score = max(0, 10 - duration_penalty)
    error_pattern = calculate_error_pattern(history)
    error_score = (error_pattern / 100) * 10
    score = pass_score + trend_score + duration_score + error_score
    flaky_cat = classify_flaky_category(history, None)
    if flaky_cat:
        penalty_map = {"timing": 8, "order_dependency": 12, "shared_state": 15, "non_deterministic_data": 20, "environment_specific": 10}
        score -= penalty_map.get(flaky_cat, 10)
    return max(0, min(100, round(score, 1)))


def calculate_score_breakdown(history):
    if not history:
        return {"pass_rate": {"value": 1.0, "weight": 0.55, "points": 55}, "recent_trend": {"value": 0, "weight": 0.20, "points": 10}, "duration_stability": {"value": 100, "weight": 0.10, "points": 10}, "error_consistency": {"value": 0, "weight": 0.10, "points": 0}, "flaky_penalty": {"category": None, "penalty": 0}, "total": 100.0, "confidence": 0.0, "bayesian_pass_rate": 0.5, "anomaly_rate": 0.0, "error_patterns": [], "sequential_pattern": None}
    bayesian_rate = calculate_bayesian_pass_rate(history)
    trend = calculate_recent_trend(history)
    duration_var = calculate_duration_variance(history)
    anomaly_rate = duration_anomaly_rate(history)
    error_pat = calculate_error_pattern(history)
    pass_score = bayesian_rate * 55
    trend_score = max(0, (50 + trend)) / 100 * 20
    duration_penalty = min(duration_var, 100) * 0.07 + anomaly_rate * 30
    duration_score = max(0, 10 - duration_penalty)
    error_score = (error_pat / 100) * 10
    total = pass_score + trend_score + duration_score + error_score
    flaky_cat = classify_flaky_category(history, None)
    penalty = 0
    if flaky_cat:
        penalty_map = {"timing": 8, "order_dependency": 12, "shared_state": 15, "non_deterministic_data": 20, "environment_specific": 10}
        penalty = penalty_map.get(flaky_cat, 10)
        total -= penalty
    total = max(0, min(100, round(total, 1)))
    confidence = calculate_score_confidence(history)
    error_patterns = get_unique_error_patterns(history)
    seq_pattern = detect_sequential_pattern(history)
    return {"pass_rate": {"value": round(bayesian_rate, 3), "weight": 0.55, "points": round(pass_score, 1)}, "recent_trend": {"value": round(trend, 1), "weight": 0.20, "points": round(trend_score, 1)}, "duration_stability": {"value": round(100 - duration_var, 1), "weight": 0.10, "points": round(duration_score, 1)}, "error_consistency": {"value": round(error_pat, 1), "weight": 0.10, "points": round(error_score, 1)}, "flaky_penalty": {"category": flaky_cat, "penalty": penalty}, "total": total, "confidence": round(confidence, 3), "bayesian_pass_rate": round(bayesian_rate, 3), "anomaly_rate": round(anomaly_rate, 3), "error_patterns": error_patterns, "sequential_pattern": seq_pattern}


def calculate_flaky_probability(history):
    if not history or len(history) < 2:
        return 0.0
    bayesian_rate = calculate_bayesian_pass_rate(history)
    if bayesian_rate < 0.05:
        return 0.0
    if bayesian_rate > 0.95:
        return 0.0
    base_prob = 1.0 - abs(bayesian_rate - 0.5) * 2
    base_prob = base_prob ** 0.8
    error_pat = calculate_error_pattern(history) / 100
    error_modifier = 1.0 + (1.0 - error_pat) * 0.3
    seq_pattern = detect_sequential_pattern(history)
    seq_modifier = 1.0
    if seq_pattern == "alternating":
        seq_modifier = 1.3
    elif seq_pattern == "clustered":
        seq_modifier = 1.15
    elif seq_pattern and seq_pattern.startswith("periodic_"):
        seq_modifier = 1.2
    anomaly = duration_anomaly_rate(history)
    dur_modifier = 1.0 + anomaly * 0.5
    confidence = calculate_score_confidence(history)
    conf_modifier = 0.5 + confidence * 0.5
    final_prob = base_prob * error_modifier * seq_modifier * dur_modifier * conf_modifier
    return min(1.0, max(0.0, round(final_prob, 3)))


# ─────────────────────────────────────────────
# Core Processing (Supabase SDK)
# ─────────────────────────────────────────────

def _ensure_repo(repo_name):
    """Ensure repository exists, return its ID."""
    existing = select_one("repositories", "id", {"name": repo_name})
    if existing:
        return existing["id"]
    result = insert("repositories", {"name": repo_name})
    return result["id"]


def process_test_run(xml_content: str, repo_name: str, run_id: str = None,
                     branch: str = "main", commit_sha: str = None,
                     environment: str = None) -> dict:
    """Main entry point: parse XML, calculate scores, store results."""
    ensure_initialized()
    logger.info("Processing test run for repo=%s run_id=%s", repo_name, run_id)

    repo_id = _ensure_repo(repo_name)
    parsed = parse_junit_xml(xml_content)

    if not run_id:
        run_id = f"run_{int(time.time())}_{hashlib.md5(xml_content.encode()).hexdigest()[:8]}"

    timestamp = datetime.now(timezone.utc).isoformat()
    total_trust = 0
    test_count = 0

    for test in parsed["tests"]:
        history = _get_test_history(repo_id, test["name"])
        
        trust_score = calculate_trust_score(history)
        confidence = calculate_score_confidence(history)
        flaky_cat = classify_flaky_category(history, test)
        flaky_prob = calculate_flaky_probability(history)

        insert("test_results", {
            "repo_id": repo_id,
            "test_name": test["name"],
            "status": test["status"],
            "duration": test["duration"],
            "error_message": test["error_message"],
            "error_type": test["error_type"],
            "flaky_category": flaky_cat,
            "run_id": run_id,
            "run_timestamp": timestamp,
            "trust_score": trust_score,
            "score_confidence": confidence,
            "environment": environment,
        })

        test["trust_score"] = trust_score
        test["score_confidence"] = confidence
        test["flaky_category"] = flaky_cat
        test["flaky_probability"] = flaky_prob
        total_trust += trust_score
        test_count += 1

    avg_trust = total_trust / test_count if test_count > 0 else 100
    
    insert("ci_runs", {
        "repo_id": repo_id,
        "run_id": run_id,
        "branch": branch,
        "commit_sha": commit_sha,
        "total_tests": parsed["total"],
        "passed": parsed["passed"],
        "failed": parsed["failed"],
        "skipped": parsed["skipped"],
        "avg_trust_score": round(avg_trust, 1),
        "timestamp": timestamp,
        "environment": environment,
    })

    logger.info("Test run processed: run_id=%s tests=%d avg_trust=%.1f", run_id, test_count, avg_trust)

    parsed["run_id"] = run_id
    parsed["avg_trust_score"] = round(avg_trust, 1)
    parsed["timestamp"] = timestamp
    return parsed


# ─────────────────────────────────────────────
# Dashboard Data (Supabase RPC)
# ─────────────────────────────────────────────

def get_dashboard_data(repo_name: str) -> dict:
    """Get aggregated data for the dashboard."""
    ensure_initialized()
    client = get_client()

    # Get stats via RPC
    stats_result = client.rpc("get_repo_stats", {"p_repo_name": repo_name}).execute()
    stats = stats_result.data[0] if stats_result.data else {}

    # Get flaky tests via RPC
    flaky_result = client.rpc("get_flaky_tests", {"p_repo_name": repo_name}).execute()
    flaky_tests = flaky_result.data if flaky_result.data else []

    # Get recent results for each flaky test
    for ft in flaky_tests:
        history_result = client.rpc("get_test_history", {
            "p_repo_name": repo_name,
            "p_test_name": ft["test_name"],
            "p_limit": 10
        }).execute()
        ft["recent_results"] = history_result.data[::-1] if history_result.data else []

    # Get recent runs
    repo = select_one("repositories", "id", {"name": repo_name})
    recent_runs = []
    if repo:
        recent_runs = select("ci_runs", "*", 
                           filters={"repo_id": repo["id"]},
                           order="-timestamp", limit=20)

    # Get trust distribution via RPC
    dist_result = client.rpc("get_trust_distribution", {"p_repo_name": repo_name}).execute()
    distribution = {r["bucket"]: r["count"] for r in dist_result.data} if dist_result.data else {}

    return {
        "stats": stats,
        "flaky_tests": flaky_tests,
        "recent_runs": recent_runs,
        "trust_distribution": distribution,
    }


def get_test_detail(repo_name: str, test_name: str) -> dict:
    """Get detailed analysis for a single test."""
    ensure_initialized()
    client = get_client()

    history_result = client.rpc("get_test_history", {
        "p_repo_name": repo_name,
        "p_test_name": test_name,
        "p_limit": 100
    }).execute()
    history = history_result.data if history_result.data else []

    trust_score = calculate_trust_score(history)
    flaky_cat = classify_flaky_category(history, None)
    flaky_prob = calculate_flaky_probability(history)
    breakdown = calculate_score_breakdown(history)

    return {
        "test_name": test_name,
        "trust_score": trust_score,
        "flaky_category": flaky_cat,
        "flaky_probability": flaky_prob,
        "total_runs": len(history),
        "pass_rate": calculate_bayesian_pass_rate(history),
        "recent_trend": calculate_recent_trend(history),
        "duration_variance": calculate_duration_variance(history),
        "error_consistency": calculate_error_pattern(history),
        "history": history[::-1],
        "breakdown": breakdown,
    }


def get_quarantined_tests(repo_name: str, threshold: float = 30) -> list:
    """Get tests that should be quarantined."""
    ensure_initialized()
    client = get_client()
    result = client.rpc("get_quarantined", {
        "p_repo_name": repo_name,
        "p_threshold": threshold
    }).execute()
    return result.data if result.data else []


# ─────────────────────────────────────────────
# Alert System
# ─────────────────────────────────────────────

def send_alert(repo_name: str, webhook_url: str, channel_type: str = "discord",
               alert_data: dict = None):
    """Send alert to Discord/Slack webhook."""
    import urllib.request

    if channel_type == "discord":
        embed = {
            "title": f"Falsky Alert — {repo_name}",
            "color": 0x8B5CF6,
            "fields": [],
            "footer": {"text": "Falsky — AI Flaky Test Trust Layer"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if alert_data:
            for key, val in alert_data.items():
                embed["fields"].append({"name": key, "value": str(val), "inline": True})
        payload = {"embeds": [embed]}
    else:
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*Falsky Alert — {repo_name}*\n" + "\n".join(f"*{k}*: {v}" for k, v in (alert_data or {}).items())}}]
        payload = {"blocks": blocks}

    try:
        req = urllib.request.Request(webhook_url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False
