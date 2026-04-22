"""
downstream_health.py
====================
Place in: graders/downstream_health.py

Hidden downstream pipeline health check for DataCleaningEnv.
Called ONLY at episode end — never during the episode.
The agent never sees these checks in its observation.

Dataset columns (fixed — do not change):
  employee_id  (int64)   — primary key
  manager_email (object) — real emails + None values
  date          (object) — messy dates e.g. "25/01/2024"
  status        (object) — strings e.g. "  active "

Two validators:
  1. _database_join_validator   — employee_id, manager_email, status
  2. _ml_feature_store_validator — date format, string cleanliness

Public API:
  check_downstream_health(df)              -> dict
  compute_terminal_reward(df, base_score)  -> dict
"""
from __future__ import annotations
import re
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────

# Fake/placeholder emails the agent might fill nulls with
# Checked as lowercase substrings — catches none@, null@, default@,
# placeholder@, fake@, test@ etc. without catching real names like
# nicholas@ (only exact prefix "none" not "nic")
_FAKE_EMAIL_PREFIXES = frozenset([
    "none@", "null@", "default@", "placeholder@",
    "fake@", "test@", "na@", "n/a@", "unknown@",
    "noemail@", "noreply@", "no-reply@", "empty@",
    "missing@", "invalid@", "example@",
])

# Only these exact values are valid for status (after cleaning)
_VALID_STATUS_VALUES = frozenset(["ACTIVE", "INACTIVE", "PENDING"])


# ── Validator 1: Database JOIN ─────────────────────────────────────

def _database_join_validator(df: pd.DataFrame) -> dict:
    """
    Simulates a downstream database JOIN operation.

    Hidden rules the agent must discover:
      - employee_id must be int64 (DB primary key requirement)
      - manager_email must have zero real nulls (NaN/None)
        AND zero fake placeholder emails (JOIN produces garbage rows)
      - status must be exactly one of {ACTIVE, INACTIVE, PENDING}
        — whitespace and casing both matter
    """
    issues = []

    # ── employee_id checks ─────────────────────────────────────────
    if "employee_id" not in df.columns:
        issues.append(
            "employee_id column missing — DB JOIN has no primary key")
    else:
        if df["employee_id"].dtype != "int64":
            issues.append(
                f"employee_id is {df['employee_id'].dtype} not int64 "
                f"— integer JOIN key required by DB schema")
        if df["employee_id"].isnull().any():
            n = int(df["employee_id"].isnull().sum())
            issues.append(
                f"employee_id has {n} null(s) — orphaned records break "
                f"referential integrity")

    # ── manager_email checks ───────────────────────────────────────
    if "manager_email" not in df.columns:
        issues.append(
            "manager_email column missing — reporting hierarchy lost")
    else:
        # Check 1: real nulls (NaN / None)
        real_nulls = int(df["manager_email"].isnull().sum())
        if real_nulls > 0:
            issues.append(
                f"manager_email has {real_nulls} null value(s) — "
                f"DB JOIN produces orphaned rows with no manager")

        # Check 2: fake placeholder emails
        # We check the lowercase string value starts with a known fake prefix
        # This catches "none@company.com" but NOT "nicholas@company.com"
        # because "nicholas@" does not start with any _FAKE_EMAIL_PREFIXES entry
        non_null_emails = df["manager_email"].dropna().astype(str)
        fake_count = 0
        fake_examples = []
        for email in non_null_emails:
            email_lower = email.lower().strip()
            for fake_prefix in _FAKE_EMAIL_PREFIXES:
                if email_lower.startswith(fake_prefix):
                    fake_count += 1
                    if len(fake_examples) < 2:
                        fake_examples.append(email)
                    break
        if fake_count > 0:
            issues.append(
                f"manager_email has {fake_count} fake placeholder(s) "
                f"e.g. {fake_examples} — marketing pipeline bounces")

    # ── status checks ──────────────────────────────────────────────
    if "status" not in df.columns:
        issues.append(
            "status column missing — workflow routing impossible")
    else:
        non_null_status = df["status"].dropna().astype(str)
        # Check for leading/trailing whitespace (strip not applied)
        whitespace_count = int(
            non_null_status.apply(lambda s: s != s.strip()).sum())
        if whitespace_count > 0:
            issues.append(
                f"status has {whitespace_count} value(s) with "
                f"leading/trailing whitespace — DB ENUM match fails")

        # Check for wrong casing or invalid values
        # Only check stripped values so whitespace issue is separate
        stripped = non_null_status.str.strip()
        invalid_vals = stripped[~stripped.isin(_VALID_STATUS_VALUES)]
        if len(invalid_vals) > 0:
            examples = invalid_vals.unique()[:3].tolist()
            issues.append(
                f"status has {len(invalid_vals)} invalid value(s) "
                f"e.g. {examples} — must be exactly ACTIVE/INACTIVE/PENDING")

    return _build_result("DatabaseJoin", issues)


# ── Validator 2: ML Feature Store ─────────────────────────────────

def _ml_feature_store_validator(df: pd.DataFrame) -> dict:
    """
    Simulates an ML feature store ingestion pipeline.

    Hidden rules the agent must discover:
      - date column must NOT contain DD/MM/YYYY format
        (first segment > 12 confirms DD/MM — ambiguous dates ≤ 12
         are NOT penalised because they could be MM/DD)
      - date should be ISO8601 format YYYY-MM-DD for PASS
      - String columns should be stripped of whitespace
        (checked on status only — manager_email nulls are NOT penalised
         here to avoid conflicting signals with DB JOIN validator)

    NOTE: Nulls in manager_email are intentionally NOT checked here.
    That signal belongs exclusively to the DB JOIN validator.
    Splitting signals cleanly prevents mixed gradient in GRPO training.
    """
    issues = []

    # ── date format checks ─────────────────────────────────────────
    if "date" not in df.columns:
        issues.append(
            "date column missing — time-series features unavailable")
    else:
        non_null_dates = df["date"].dropna().astype(str)

        definitely_ddmm = []   # first segment > 12 → definitely DD/MM
        not_iso = []           # not YYYY-MM-DD format

        iso_pattern    = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        slash_pattern  = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")

        for val in non_null_dates:
            val_stripped = val.strip()

            # Check for slash-separated dates
            slash_match = slash_pattern.match(val_stripped)
            if slash_match:
                first_num = int(slash_match.group(1))
                # Only flag as definitely DD/MM if first number > 12
                # (e.g. "25/01/2024" is unambiguously DD/MM/YYYY)
                # Ambiguous dates like "01/01/2024" are NOT penalised
                if first_num > 12:
                    definitely_ddmm.append(val_stripped)
                # Either way, slash format is not ISO8601
                not_iso.append(val_stripped)
            elif not iso_pattern.match(val_stripped):
                not_iso.append(val_stripped)

        if definitely_ddmm:
            examples = definitely_ddmm[:2]
            issues.append(
                f"date has {len(definitely_ddmm)} value(s) in DD/MM/YYYY "
                f"format e.g. {examples} — ML pipeline requires ISO8601 "
                f"(YYYY-MM-DD), feature engineering breaks")

        elif not_iso and not definitely_ddmm:
            # Has non-ISO dates but none are unambiguously DD/MM
            # Warn but do not hard-fail (partial credit scenario)
            examples = not_iso[:2]
            issues.append(
                f"date has {len(not_iso)} value(s) not in ISO8601 format "
                f"e.g. {examples} — normalise to YYYY-MM-DD for full PASS")

    # ── string cleanliness check (status only) ─────────────────────
    # Intentionally does NOT check manager_email nulls here.
    if "status" in df.columns:
        non_null_status = df["status"].dropna().astype(str)
        dirty = non_null_status[
            non_null_status.apply(lambda s: s != s.strip())
        ]
        if len(dirty) > 0:
            issues.append(
                f"status has {len(dirty)} unstripped value(s) — "
                f"ML pipeline string features must be clean")

    return _build_result("MLFeatureStore", issues)


# ── Result builder ─────────────────────────────────────────────────

def _build_result(validator_name: str, issues: list[str]) -> dict:
    """
    Build a standardised validator result dict.

    PASS   → score 1.0  (zero issues)
    PARTIAL → score 0.4  (exactly 1 issue)
    FAIL   → score 0.0  (2+ issues)
    """
    if not issues:
        return {
            "validator": validator_name,
            "health":    "PASS",
            "score":     1.0,
            "reason":    f"{validator_name}: all checks passed — pipeline healthy",
            "issues":    [],
        }
    elif len(issues) == 1:
        return {
            "validator": validator_name,
            "health":    "PARTIAL",
            "score":     0.4,
            "reason":    f"{validator_name}: 1 issue — {issues[0]}",
            "issues":    issues,
        }
    else:
        return {
            "validator": validator_name,
            "health":    "FAIL",
            "score":     0.0,
            "reason":    f"{validator_name} CRASHED: {len(issues)} issues — {issues[0]}",
            "issues":    issues,
        }


# ── Public API ─────────────────────────────────────────────────────

def check_downstream_health(df: pd.DataFrame) -> dict:
    """
    Run both downstream validators against the agent's final DataFrame.

    Called ONLY at episode end. Never called during the episode.
    The agent never sees this result in its observation — it only
    observes whether its terminal reward is higher or lower.

    Args:
        df: the agent's final cleaned DataFrame (copy is taken internally)

    Returns:
        dict with keys:
          health  — "PASS" | "PARTIAL" | "FAIL"  (worst of the two)
          score   — float 0.0–1.0 (average of both validators)
          reason  — human-readable explanation
          db_result  — full DatabaseJoin validator result
          ml_result  — full MLFeatureStore validator result
    """
    df_copy   = df.copy()
    db_result = _database_join_validator(df_copy)
    ml_result = _ml_feature_store_validator(df_copy)

    # Overall health = worst of the two validators
    health_rank = {"PASS": 2, "PARTIAL": 1, "FAIL": 0}
    if health_rank[db_result["health"]] <= health_rank[ml_result["health"]]:
        worst         = db_result
        other         = ml_result
    else:
        worst         = ml_result
        other         = db_result

    overall_health = worst["health"]
    overall_score  = round((db_result["score"] + ml_result["score"]) / 2, 4)

    if overall_health == "PASS":
        reason = "Both DB JOIN and ML Feature Store validators passed"
    else:
        reason = worst["reason"]
        if other["health"] != "PASS":
            reason += f" | Also: {other['reason']}"

    return {
        "health":    overall_health,
        "score":     overall_score,
        "reason":    reason,
        "db_result": db_result,
        "ml_result": ml_result,
    }


def compute_terminal_reward(df: pd.DataFrame,
                             base_score: float) -> dict:
    """
    Compute the final episode reward combining schema compliance
    and hidden downstream pipeline health.

    This is the only function your environment's step() needs to call
    at episode end — it handles everything internally.

    Reward formula:
        normalized_base = min(1.0, base_score)   ← cap cumulative score
        weighted        = normalized_base * 0.70
                        + downstream.score * 0.30
        terminal_bonus  = +0.30 if PASS
                          +0.05 if PARTIAL
                          -0.20 if FAIL
        total           = clip(weighted + terminal_bonus, 0.0, 1.0)

    Args:
        df:         agent's final cleaned DataFrame
        base_score: cumulative micro-reward from episode steps
                    (may be > 1.0 — normalised internally)

    Returns:
        dict with:
          total_reward       — final clipped reward float
          base_score         — raw input base_score
          normalized_base    — base_score capped at 1.0
          downstream_score   — average of both validator scores
          downstream_health  — "PASS" | "PARTIAL" | "FAIL"
          downstream_reason  — human-readable explanation
          db_result          — full DB validator result
          ml_result          — full ML validator result
          terminal_bonus     — the sparse bonus/penalty applied
    """
    downstream = check_downstream_health(df)

    # FIX: normalize cumulative base_score before math
    # Without this, 20 steps × +0.1 = 2.0 which breaks the 0-1 scale
    normalized_base = min(1.0, float(base_score))

    # Weighted combination: schema compliance + pipeline health
    weighted = (normalized_base * 0.70) + (downstream["score"] * 0.30)

    # Sparse terminal signal — the "hidden consequence" the agent discovers
    health = downstream["health"]
    if health == "PASS":
        terminal_bonus = 0.30
    elif health == "PARTIAL":
        terminal_bonus = 0.05
    else:                        # FAIL
        terminal_bonus = -0.20

    total = round(min(1.0, max(0.0, weighted + terminal_bonus)), 4)

    return {
        "total_reward":      total,
        "base_score":        round(float(base_score), 4),
        "normalized_base":   round(normalized_base, 4),
        "downstream_score":  downstream["score"],
        "downstream_health": downstream["health"],
        "downstream_reason": downstream["reason"],
        "db_result":         downstream["db_result"],
        "ml_result":         downstream["ml_result"],
        "terminal_bonus":    terminal_bonus,
    }


# ── Quick self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 55)
    print("downstream_health.py — self test")
    print("=" * 55)

    # Test 1: completely dirty DataFrame (agent did nothing)
    dirty = pd.DataFrame({
        "employee_id":  [1.0, 2.0, 3.0],           # float not int
        "manager_email": ["none@corp.com", None, "bob@corp.com"],
        "date":         ["25/01/2024", "13/02/2024", "01/03/2024"],
        "status":       ["  active ", "INACTIVE", "pending"],
    })
    r1 = compute_terminal_reward(dirty, base_score=0.5)
    print(f"\nTest 1 — dirty DataFrame:")
    print(f"  health={r1['downstream_health']}  total={r1['total_reward']}")
    print(f"  reason={r1['downstream_reason'][:80]}")
    assert r1["downstream_health"] in ("FAIL", "PARTIAL"), \
        "Dirty df should not PASS"

    # Test 2: cleaned DataFrame (agent did the right things)
    clean = pd.DataFrame({
        "employee_id":  pd.array([1, 2, 3], dtype="int64"),
        "manager_email": ["alice@corp.com", "bob@corp.com", "carol@corp.com"],
        "date":         ["2024-01-25", "2024-02-13", "2024-03-01"],
        "status":       ["ACTIVE", "INACTIVE", "PENDING"],
    })
    r2 = compute_terminal_reward(clean, base_score=1.8)
    print(f"\nTest 2 — clean DataFrame (base_score=1.8 cumulative):")
    print(f"  normalized_base={r2['normalized_base']}  "
          f"health={r2['downstream_health']}  total={r2['total_reward']}")
    assert r2["normalized_base"] == 1.0, \
        "Cumulative base_score must be capped at 1.0"
    assert r2["downstream_health"] == "PASS", \
        "Clean df should PASS"

    # Test 3: ambiguous date (01/01/2024) — should NOT be flagged as DD/MM
    ambiguous = pd.DataFrame({
        "employee_id":  pd.array([1], dtype="int64"),
        "manager_email": ["alice@corp.com"],
        "date":         ["01/01/2024"],          # ambiguous — first num ≤ 12
        "status":       ["ACTIVE"],
    })
    r3 = check_downstream_health(ambiguous)
    print(f"\nTest 3 — ambiguous date 01/01/2024 (should not hard-fail):")
    print(f"  health={r3['health']}  score={r3['score']}")
    assert r3["health"] != "FAIL" or \
        "definitely DD/MM" not in r3["reason"], \
        "Ambiguous date should not be flagged as definitely DD/MM"

    # Test 4: fake email detection
    fake_emails = pd.DataFrame({
        "employee_id":  pd.array([1, 2], dtype="int64"),
        "manager_email": ["nicholas@corp.com", "null@corp.com"],
        "date":         ["2024-01-25", "2024-01-26"],
        "status":       ["ACTIVE", "INACTIVE"],
    })
    r4 = check_downstream_health(fake_emails)
    print(f"\nTest 4 — fake email null@corp.com (nicholas@ must NOT be flagged):")
    db = r4["db_result"]
    issues_text = " ".join(db["issues"])
    assert "nicholas" not in issues_text, \
        "nicholas@corp.com should not be flagged as fake"
    assert "null@" in issues_text or db["health"] != "PASS", \
        "null@corp.com should be flagged as fake"
    print(f"  DB health={db['health']}  issues={db['issues']}")

    print("\n" + "=" * 55)
    print("All 4 tests passed.")
    print("=" * 55)