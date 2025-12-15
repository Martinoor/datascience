"""Small helper to submit predictions to Kaggle and track the result.

This keeps all submission-related code in one place so notebooks can call
``submit_and_track`` to push a file, poll for the public score, and append a
line to ``submission_log.csv`` with useful metadata.

Usage example (after writing submission.csv):

>>> from kaggle_submit import submit_and_track
>>> submit_and_track("submission.csv", "my-competition", "run-001")

You need a valid Kaggle API token at ``~/.kaggle/kaggle.json``.
"""
#%%
from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Any

from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"


def _ensure_token() -> None:
    """Ensure the Kaggle API token exists and has safe permissions."""

    if not KAGGLE_TOKEN_PATH.exists():
        raise FileNotFoundError(
            "Missing ~/.kaggle/kaggle.json; download from Kaggle > Account > API."
        )
    # Kaggle API checks for 600 permissions on the token file.
    KAGGLE_TOKEN_PATH.chmod(0o600)


def _safe_get(obj: Any, keys: Iterable[str]) -> Optional[Any]:
    """Try multiple attribute/dict keys; return the first non-None value."""
    for key in keys:
        # attribute
        if hasattr(obj, key):
            try:
                v = getattr(obj, key)
            except Exception:
                v = None
            if v is not None:
                return v
        # dict
        if isinstance(obj, dict):
            v = obj.get(key)
            if v is not None:
                return v
    return None
def _status_to_str(s: Any) -> str:
    # `_status` is a SubmissionStatus enum
    if s is None:
        return ""
    return str(getattr(s, "name", s)).lower()

def _submission_to_dict(submission: object) -> Dict[str, object]:
    """Normalize Kaggle submission objects to a plain dict."""

    data = {
    "ref": _safe_get(submission, ["ref", "_ref"]),
    "description": _safe_get(submission, ["description", "_description"]),
    "status": _status_to_str(_safe_get(submission, ["status", "_status"])),
    "score": _safe_get(submission, ["publicScore", "public_score", "_public_score", "score"]),
    "fileName": _safe_get(submission, ["fileName", "file_name", "_file_name"]),
    "submissionDate": _safe_get(submission, ["submissionDate", "date", "_date"]),
    "error": _safe_get(submission, ["errorDescription", "error_description", "_error_description"]),
    "teamName": _safe_get(submission, ["teamName", "team_name", "_team_name"]),
    "submittedBy": _safe_get(submission, ["submittedBy", "submitted_by", "_submitted_by"]),
    "isFrozen": _safe_get(submission, ["isFrozen", "is_frozen", "_is_frozen"]),
    }

    # Kaggle sometimes reports scores as strings; try to coerce.
    try:
        if data["score"] is not None:
            data["score"] = float(data["score"])
    except Exception:
        pass

    # Treat empty strings or NaN as missing scores so polling can continue.
    score = data.get("score")
    if score == "" or (isinstance(score, float) and math.isnan(score)):
        data["score"] = None

    return data


def _latest_submission(
    submissions: Iterable[object], description: str | None = None
) -> Optional[Dict[str, object]]:
    """Return the most recent submission info from the Kaggle API list.

    If ``description`` is provided, prefer the newest submission whose description
    matches; otherwise fall back to the newest overall.
    """

    submissions = list(submissions)
    if not submissions:
        return None

    def sort_key(item: object):
        # Kaggle submissions have monotonically increasing refs and timestamps.
        dt = _safe_get(item, ["submissionDate", "date", "_date"])
        ref = _safe_get(item, ["ref", "_ref"]) or 0
        return (dt is not None, dt, ref)

    def newest(items: list[object]) -> object:
        # max() works with the tuple sort_key above.
        return max(items, key=sort_key)

    if description:
        matches = [obj for obj in submissions if _safe_get(obj, ["description", "_description"]) == description]
        if matches:
            return _submission_to_dict(newest(matches))

    return _submission_to_dict(newest(submissions))


def _append_log(log_path: Path, row: Dict[str, object]) -> None:
    """Append a row to the submission log CSV, creating it if missing."""

    fieldnames = list(row.keys())
    needs_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def submit_and_track(
    submission_path: str | Path,
    competition: str,
    message: str,
    *,
    wait_for_result: bool = True,
    poll_interval: int = 19,
    max_polls: int = 8,
    log_path: str | Path | None = "submission_log.csv",
    extra_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Submit to Kaggle, optionally poll for a score, and log metadata.

    Parameters
    ----------
    submission_path:
        Path to the CSV to upload.
    competition:
        Kaggle competition slug, e.g. ``"churn-prediction-25-26"``.
    message:
        Submission description that shows up on Kaggle.
    wait_for_result:
        If True, poll the submissions endpoint for a public score.
    poll_interval:
        Seconds to sleep between polls.
    max_polls:
        Maximum number of polls before giving up.
    log_path:
        Where to append results. Set to None to skip logging.
    extra_meta:
        Additional columns to include in the log (e.g., thresholds, metrics).
    """

    submission_path = Path(submission_path)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    _ensure_token()

    api = KaggleApi()
    api.authenticate()

    print(f"Submitting {submission_path} to {competition} with note: {message}")
    response = api.competition_submit(submission_path, message, competition)
    info = _submission_to_dict(response)
    info.setdefault("description", message)
    info.setdefault("fileName", submission_path.name)
    info.setdefault("status", "pending")

    if wait_for_result:
        missing_score_polls = 0
        for i in range(max_polls):
            time.sleep(poll_interval)
            submissions = api.competition_submissions(competition)
            latest = _latest_submission(submissions, description=message)
            if latest:
                info.update(latest)

                # Optional: log if the latest submission description differs from the request.
                if (
                    latest.get("description")
                    and message
                    and latest["description"] != message
                ):
                    print(
                        "Latest submission description differs from requested message; "
                        "continuing to poll in case a newer one appears."
                    )

            print(
                f"Poll {i + 1}/{max_polls}: status={info.get('status')} score={info.get('score')}"
            )

            status = info.get("status")
            has_score =  (
                            info.get("score") is not None
                            and info.get("score") != ''
                        )

            # Keep polling for up to 3 extra iterations when status is complete but score is missing.
            if status and status != "pending":
                if has_score:
                    break
                missing_score_polls += 1
                if missing_score_polls >= 3:
                    break

    log_record = {
        "logged_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        **info,
    }
    if extra_meta:
        log_record.update(extra_meta)

    if log_path:
        _append_log(Path(log_path), log_record)

    return info

# %%
