"""
Falsky — Real pytest tests for the core engine and API.
Includes stable tests, intentionally flaky tests, and edge-case tests
to demonstrate Falsky's flaky detection capability on live data.
"""

import math
import time
import os
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# STABLE TESTS — These should always pass
# ─────────────────────────────────────────────

class TestTrustEngine:
    """Tests for the core Bayesian trust engine."""

    def test_parse_junit_basic(self):
        """Parse a basic JUnit XML with one pass and one fail."""
        xml = """<?xml version="1.0"?>
        <testsuites>
          <testsuite name="suite1" tests="2" failures="1">
            <testcase name="test_ok" classname="Demo" time="0.12"/>
            <testcase name="test_fail" classname="Demo" time="0.35">
              <failure message="AssertionError">expected True got False</failure>
            </testcase>
          </testsuite>
        </testsuites>"""
        from engine.trust_engine import parse_junit_xml
        result = parse_junit_xml(xml)
        assert result["total"] == 2
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["tests"][0]["status"] == "passed"
        assert result["tests"][1]["status"] == "failed"

    def test_parse_junit_skipped(self):
        """Parse JUnit XML with a skipped test."""
        xml = """<?xml version="1.0"?>
        <testsuites>
          <testsuite name="suite1" tests="2" failures="0">
            <testcase name="test_run" classname="Demo" time="0.05"/>
            <testcase name="test_skip" classname="Demo" time="0.0">
              <skipped/>
            </testcase>
          </testsuite>
        </testsuites>"""
        from engine.trust_engine import parse_junit_xml
        result = parse_junit_xml(xml)
        assert result["skipped"] == 1
        assert result["passed"] == 1

    def test_parse_junit_empty(self):
        """Parse empty/invalid XML gracefully."""
        from engine.trust_engine import parse_junit_xml
        result = parse_junit_xml("<invalid>")
        assert result["total"] == 0
        assert result["tests"] == []

    def test_parse_junit_error_element(self):
        """Parse JUnit XML with <error> element (not just <failure>)."""
        xml = """<?xml version="1.0"?>
        <testsuites>
          <testsuite name="suite1" tests="1" errors="1">
            <testcase name="test_crash" classname="Demo" time="0.01">
              <error message="RuntimeError">process crashed</error>
            </testcase>
          </testsuite>
        </testsuites>"""
        from engine.trust_engine import parse_junit_xml
        result = parse_junit_xml(xml)
        assert result["failed"] == 1
        assert result["tests"][0]["error_type"] == "error"

    def test_bayesian_pass_rate(self):
        """Bayesian pass rate with known history."""
        from engine.trust_engine import calculate_bayesian_pass_rate
        # 8/10 passed — MAP estimate with Beta(2,2) prior
        history = [{"status": "passed"} for _ in range(8)] + [{"status": "failed"} for _ in range(2)]
        rate = calculate_bayesian_pass_rate(history, prior_alpha=2.0, prior_beta=2.0)
        # alpha=10, beta=4, MAP = (10-1)/(10+4-2) = 9/12 = 0.75
        assert rate == 0.75

    def test_bayesian_pass_rate_no_history(self):
        """No history returns prior mean (0.5)."""
        from engine.trust_engine import calculate_bayesian_pass_rate
        rate = calculate_bayesian_pass_rate([], prior_alpha=2.0, prior_beta=2.0)
        assert rate == 0.5

    def test_bayesian_pass_rate_all_pass(self):
        """All passes with enough runs → high rate but not exactly 1.0."""
        from engine.trust_engine import calculate_bayesian_pass_rate
        history = [{"status": "passed"} for _ in range(20)]
        rate = calculate_bayesian_pass_rate(history, prior_alpha=2.0, prior_beta=2.0)
        # alpha=22, beta=2, MAP = 21/22 ≈ 0.954
        assert 0.9 < rate < 1.0

    def test_test_hash_consistency(self):
        """Test hash is deterministic for same inputs."""
        from engine.trust_engine import _test_hash
        h1 = _test_hash("my_test", "my_repo")
        h2 = _test_hash("my_test", "my_repo")
        assert h1 == h2
        assert len(h1) == 16

    def test_test_hash_different_inputs(self):
        """Different inputs produce different hashes."""
        from engine.trust_engine import _test_hash
        h1 = _test_hash("test_a", "repo_x")
        h2 = _test_hash("test_b", "repo_x")
        assert h1 != h2

    def test_beta_pdf_bounds(self):
        """Beta PDF returns 0 at boundaries."""
        from engine.trust_engine import _beta_pdf
        assert _beta_pdf(0, 2, 2) == 0.0
        assert _beta_pdf(1, 2, 2) == 0.0
        assert _beta_pdf(0.5, 2, 2) > 0.0


class TestAPIBasic:
    """Tests for FastAPI endpoints (no auth required)."""

    def test_health_endpoint(self):
        """Health endpoint returns ok status."""
        from backend.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "3.0.0"

    def test_robots_txt(self):
        """Robots.txt endpoint returns plain text."""
        from backend.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/robots.txt")
        assert resp.status_code == 200
        assert "User-agent" in resp.text

    def test_favicon(self):
        """Favicon returns SVG image."""
        from backend.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/favicon.ico")
        assert resp.status_code == 200
        assert "svg" in resp.text.lower()


class TestUtilities:
    """Utility function tests."""

    def test_make_junit_xml_roundtrip(self):
        """Generate JUnit XML → parse it back → verify counts."""
        root = ET.Element("testsuites")
        suite = ET.SubElement(root, "testsuite", name="rt", tests="3")
        ET.SubElement(suite, "testcase", name="t1", classname="C", time="0.1")
        tc2 = ET.SubElement(suite, "testcase", name="t2", classname="C", time="0.2")
        ET.SubElement(tc2, "failure", message="err")
        ET.SubElement(suite, "testcase", name="t3", classname="C", time="0.3")
        xml = ET.tostring(root, encoding="unicode")

        from engine.trust_engine import parse_junit_xml
        result = parse_junit_xml(xml)
        assert result["total"] == 3
        assert result["passed"] == 2
        assert result["failed"] == 1


# ─────────────────────────────────────────────
# INTENTIONALLY FLAKY TESTS — These demonstrate Falsky's detection
# ─────────────────────────────────────────────

class TestFlakyTiming:
    """Flaky due to timing — passes if response is fast, fails if slow."""

    def test_api_timeout_sensitive(self):
        """This test is flaky: depends on timing threshold.
        ~60% pass rate expected across multiple CI runs."""
        start = time.time()
        # Simulate variable latency
        run_num = int(os.environ.get("CI_RUN_NUMBER", "0") or "0")
        delay = 0.05 if run_num % 3 != 0 else 0.5
        time.sleep(delay)
        elapsed = time.time() - start
        # Threshold: 0.2s — sometimes passes, sometimes fails
        assert elapsed < 0.2, f"Response took {elapsed:.3f}s, threshold 0.2s"


class TestFlakySharedState:
    """Flaky due to shared mutable state — order-dependent."""

    _shared_counter = 0

    def test_increment_shared_counter(self):
        """This test is flaky: depends on shared state that other tests modify.
        Passes if counter is in expected range, fails if contaminated."""
        # Reset sometimes, sometimes don't — creates order dependency
        run_num = int(os.environ.get("CI_RUN_NUMBER", "0") or "0")
        if run_num % 4 == 0:
            TestFlakySharedState._shared_counter = 0
        TestFlakySharedState._shared_counter += 1
        # Passes only if counter is 1 (fresh start), fails if already incremented
        assert TestFlakySharedState._shared_counter == 1, \
            f"Counter was {TestFlakySharedState._shared_counter}, expected 1 (shared state leak)"


class TestFlakyEnvironment:
    """Flaky due to environment differences — passes locally, may fail in CI."""

    def test_environment_specific_path(self):
        """This test is flaky: uses /tmp which behaves differently on some systems.
        ~70% pass rate expected."""
        test_dir = "/tmp/falsky_test_" + str(int(time.time()))
        # Sometimes this fails due to permissions or existing state
        try:
            os.makedirs(test_dir, exist_ok=True)
            # Write and read back
            with open(os.path.join(test_dir, "data.txt"), "w") as f:
                f.write("test")
            with open(os.path.join(test_dir, "data.txt"), "r") as f:
                content = f.read()
            assert content == "test"
            # Clean up — but sometimes fails on CI
            os.remove(os.path.join(test_dir, "data.txt"))
            os.rmdir(test_dir)
        except OSError as e:
            # Intentionally fail sometimes on CI
            run_num = int(os.environ.get("CI_RUN_NUMBER", "0") or "0")
            if os.environ.get("CI") == "true" and run_num % 5 == 0:
                assert False, f"Environment flake: {e}"


class TestFlakyNonDeterministic:
    """Flaky due to random/non-deterministic data."""

    def test_random_data_sorting(self):
        """This test is flaky: random data sometimes produces unexpected order.
        ~50% pass rate — the most unreliable test."""
        import random
        random.seed()  # No fixed seed — truly random
        data = [random.randint(1, 100) for _ in range(10)]
        sorted_data = sorted(data)
        # Check that first element is smallest — but random data can be tricky
        # This will fail about 50% of the time with the assertion below
        # because we assert something that's sometimes false with random data
        median = sorted_data[len(sorted_data) // 2]
        # The median of 10 random ints 1-100 is usually > 30
        # But sometimes it's not — that's the flake
        assert median > 30, f"Median {median} too low — non-deterministic data issue"


# ─────────────────────────────────────────────
# BROKEN TEST — Always fails, not flaky
# ─────────────────────────────────────────────

class TestBroken:
    """Always fails — broken test, not flaky."""

    def test_deprecated_endpoint_removed(self):
        """This endpoint no longer exists — test is broken, needs update."""
        # This will always fail — the endpoint was removed
        assert False, "Endpoint /api/v1/deprecated removed — test needs update"
