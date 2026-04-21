"""Tests for cron/jobs.py — schedule parsing, job CRUD, and due-job detection."""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cron.jobs import (
    parse_duration,
    parse_schedule,
    compute_next_run,
    create_job,
    load_jobs,
    save_jobs,
    get_job,
    list_jobs,
    update_job,
    pause_job,
    resume_job,
    remove_job,
    mark_job_run,
    advance_next_run,
    get_due_jobs,
    save_job_output,
)


# =========================================================================
# parse_duration
# =========================================================================

class TestParseDuration:
    def test_minutes(self):
        assert parse_duration("30m") == 30
        assert parse_duration("1min") == 1
        assert parse_duration("5mins") == 5
        assert parse_duration("10minute") == 10
        assert parse_duration("120minutes") == 120

    def test_hours(self):
        assert parse_duration("2h") == 120
        assert parse_duration("1hr") == 60
        assert parse_duration("3hrs") == 180
        assert parse_duration("1hour") == 60
        assert parse_duration("24hours") == 1440

    def test_days(self):
        assert parse_duration("1d") == 1440
        assert parse_duration("7day") == 7 * 1440
        assert parse_duration("2days") == 2 * 1440

    def test_whitespace_tolerance(self):
        assert parse_duration("  30m  ") == 30
        assert parse_duration("2 h") == 120

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("30x")
        with pytest.raises(ValueError):
            parse_duration("")
        with pytest.raises(ValueError):
            parse_duration("m30")


# =========================================================================
# parse_schedule
# =========================================================================

class TestParseSchedule:
    def test_duration_becomes_once(self):
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert "run_at" in result
        # run_at should be a valid ISO timestamp string ~30 minutes from now
        run_at_str = result["run_at"]
        assert isinstance(run_at_str, str)
        run_at = datetime.fromisoformat(run_at_str)
        now = datetime.now().astimezone()
        assert run_at > now
        assert run_at < now + timedelta(minutes=31)

    def test_every_becomes_interval(self):
        result = parse_schedule("every 2h")
        assert result["kind"] == "interval"
        assert result["minutes"] == 120

    def test_every_case_insensitive(self):
        result = parse_schedule("Every 30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30

    def test_cron_expression(self):
        pytest.importorskip("croniter")
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"

    def test_iso_timestamp(self):
        result = parse_schedule("2030-01-15T14:00:00")
        assert result["kind"] == "once"
        assert "2030-01-15" in result["run_at"]

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("not_a_schedule")

    def test_invalid_cron_raises(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            parse_schedule("99 99 99 99 99")


# =========================================================================
# compute_next_run
# =========================================================================

class TestComputeNextRun:
    def test_once_future_returns_time(self):
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": future}
        assert compute_next_run(schedule) == future

    def test_once_recent_past_within_grace_returns_time(self, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 3, tzinfo=timezone.utc)
        run_at = "2026-03-18T04:22:00+00:00"
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "once", "run_at": run_at}

        assert compute_next_run(schedule) == run_at

    def test_once_past_returns_none(self):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": past}
        assert compute_next_run(schedule) is None

    def test_once_with_last_run_returns_none_even_within_grace(self, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 3, tzinfo=timezone.utc)
        run_at = "2026-03-18T04:22:00+00:00"
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "once", "run_at": run_at}

        assert compute_next_run(schedule, last_run_at=now.isoformat()) is None

    def test_interval_first_run(self):
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        next_dt = datetime.fromisoformat(result)
        # Should be ~60 minutes from now
        assert next_dt > datetime.now().astimezone() + timedelta(minutes=59)

    def test_interval_subsequent_run(self):
        schedule = {"kind": "interval", "minutes": 30}
        last = datetime.now().astimezone().isoformat()
        result = compute_next_run(schedule, last_run_at=last)
        next_dt = datetime.fromisoformat(result)
        # Should be ~30 minutes from last run
        assert next_dt > datetime.now().astimezone() + timedelta(minutes=29)

    def test_cron_returns_future(self):
        pytest.importorskip("croniter")
        schedule = {"kind": "cron", "expr": "* * * * *"}  # every minute
        result = compute_next_run(schedule)
        assert isinstance(result, str), f"Expected ISO timestamp string, got {type(result)}"
        assert len(result) > 0
        next_dt = datetime.fromisoformat(result)
        assert isinstance(next_dt, datetime)
        assert next_dt > datetime.now().astimezone()

    def test_unknown_kind_returns_none(self):
        assert compute_next_run({"kind": "unknown"}) is None


# =========================================================================
# Job CRUD (with tmp file storage)
# =========================================================================

@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs.JOBS_LOCK_FILE", tmp_path / "cron" / ".jobs.lock")
    return tmp_path


class TestJobCRUD:
    def test_create_and_get(self, tmp_cron_dir):
        job = create_job(prompt="Check server status", schedule="30m")
        assert job["id"]
        assert job["prompt"] == "Check server status"
        assert job["enabled"] is True
        assert job["schedule"]["kind"] == "once"

        fetched = get_job(job["id"])
        assert fetched is not None
        assert fetched["prompt"] == "Check server status"

    def test_list_jobs(self, tmp_cron_dir):
        create_job(prompt="Job 1", schedule="every 1h")
        create_job(prompt="Job 2", schedule="every 2h")
        jobs = list_jobs()
        assert len(jobs) == 2

    def test_remove_job(self, tmp_cron_dir):
        job = create_job(prompt="Temp job", schedule="30m")
        assert remove_job(job["id"]) is True
        assert get_job(job["id"]) is None

    def test_remove_nonexistent_returns_false(self, tmp_cron_dir):
        assert remove_job("nonexistent") is False

    def test_auto_repeat_for_once(self, tmp_cron_dir):
        job = create_job(prompt="One-shot", schedule="1h")
        assert job["repeat"]["times"] == 1

    def test_interval_no_auto_repeat(self, tmp_cron_dir):
        job = create_job(prompt="Recurring", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_default_delivery_origin(self, tmp_cron_dir):
        job = create_job(
            prompt="Test", schedule="30m",
            origin={"platform": "telegram", "chat_id": "123"},
        )
        assert job["deliver"] == "origin"

    def test_default_delivery_local_no_origin(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="30m")
        assert job["deliver"] == "local"


class TestUpdateJob:
    def test_update_name(self, tmp_cron_dir):
        job = create_job(prompt="Check server status", schedule="every 1h", name="Old Name")
        assert job["name"] == "Old Name"
        updated = update_job(job["id"], {"name": "New Name"})
        assert updated is not None
        assert isinstance(updated, dict)
        assert updated["name"] == "New Name"
        # Verify other fields are preserved
        assert updated["prompt"] == "Check server status"
        assert updated["id"] == job["id"]
        assert updated["schedule"] == job["schedule"]
        # Verify persisted to disk
        fetched = get_job(job["id"])
        assert fetched["name"] == "New Name"

    def test_update_schedule(self, tmp_cron_dir):
        job = create_job(prompt="Daily report", schedule="every 1h")
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 60
        old_next_run = job["next_run_at"]
        new_schedule = parse_schedule("every 2h")
        updated = update_job(job["id"], {"schedule": new_schedule, "schedule_display": new_schedule["display"]})
        assert updated is not None
        assert updated["schedule"]["kind"] == "interval"
        assert updated["schedule"]["minutes"] == 120
        assert updated["schedule_display"] == "every 120m"
        assert updated["next_run_at"] != old_next_run
        # Verify persisted to disk
        fetched = get_job(job["id"])
        assert fetched["schedule"]["minutes"] == 120
        assert fetched["schedule_display"] == "every 120m"

    def test_update_enable_disable(self, tmp_cron_dir):
        job = create_job(prompt="Toggle me", schedule="every 1h")
        assert job["enabled"] is True
        updated = update_job(job["id"], {"enabled": False})
        assert updated["enabled"] is False
        fetched = get_job(job["id"])
        assert fetched["enabled"] is False

    def test_update_nonexistent_returns_none(self, tmp_cron_dir):
        result = update_job("nonexistent_id", {"name": "X"})
        assert result is None


class TestPauseResumeJob:
    def test_pause_sets_state(self, tmp_cron_dir):
        job = create_job(prompt="Pause me", schedule="every 1h")
        paused = pause_job(job["id"], reason="user paused")
        assert paused is not None
        assert paused["enabled"] is False
        assert paused["state"] == "paused"
        assert paused["paused_reason"] == "user paused"

    def test_resume_reenables_job(self, tmp_cron_dir):
        job = create_job(prompt="Resume me", schedule="every 1h")
        pause_job(job["id"], reason="user paused")
        resumed = resume_job(job["id"])
        assert resumed is not None
        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["paused_at"] is None
        assert resumed["paused_reason"] is None


class TestMarkJobRun:
    def test_increments_completed(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="every 1h")
        mark_job_run(job["id"], success=True)
        updated = get_job(job["id"])
        assert updated["repeat"]["completed"] == 1
        assert updated["last_status"] == "ok"

    def test_repeat_limit_removes_job(self, tmp_cron_dir):
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True)
        # Job should be removed after hitting repeat limit
        assert get_job(job["id"]) is None

    def test_repeat_negative_one_is_infinite(self, tmp_cron_dir):
        # LLMs often pass repeat=-1 to mean "infinite/forever".
        # The job must NOT be deleted after runs when repeat <= 0.
        job = create_job(prompt="Forever", schedule="every 1h", repeat=-1)
        # -1 should be normalised to None (infinite) at create time
        assert job["repeat"]["times"] is None
        # Running it multiple times should never delete it
        for _ in range(3):
            mark_job_run(job["id"], success=True)
            assert get_job(job["id"]) is not None, "job was deleted after run despite infinite repeat"

    def test_repeat_zero_is_infinite(self, tmp_cron_dir):
        # repeat=0 should also be treated as None (infinite), not "run zero times".
        job = create_job(prompt="ZeroRepeat", schedule="every 1h", repeat=0)
        assert job["repeat"]["times"] is None
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"]) is not None

    def test_error_status(self, tmp_cron_dir):
        job = create_job(prompt="Fail", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="timeout")
        updated = get_job(job["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "timeout"

    def test_delivery_error_tracked_separately(self, tmp_cron_dir):
        """Agent succeeds but delivery fails — both tracked independently."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=True, delivery_error="platform 'telegram' not configured")
        updated = get_job(job["id"])
        assert updated["last_status"] == "ok"
        assert updated["last_error"] is None
        assert updated["last_delivery_error"] == "platform 'telegram' not configured"

    def test_delivery_error_cleared_on_success(self, tmp_cron_dir):
        """Successful delivery clears the previous delivery error."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=True, delivery_error="network timeout")
        updated = get_job(job["id"])
        assert updated["last_delivery_error"] == "network timeout"
        # Next run delivers successfully
        mark_job_run(job["id"], success=True, delivery_error=None)
        updated = get_job(job["id"])
        assert updated["last_delivery_error"] is None

    def test_both_agent_and_delivery_error(self, tmp_cron_dir):
        """Agent fails AND delivery fails — both errors recorded."""
        job = create_job(prompt="Report", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="model timeout",
                     delivery_error="platform 'discord' not enabled")
        updated = get_job(job["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "model timeout"
        assert updated["last_delivery_error"] == "platform 'discord' not enabled"


class TestAdvanceNextRun:
    """Tests for advance_next_run() — crash-safety for recurring jobs."""

    def test_advances_interval_job(self, tmp_cron_dir):
        """Interval jobs should have next_run_at bumped to the next future occurrence."""
        job = create_job(prompt="Recurring check", schedule="every 1h")
        # Force next_run_at to 5 minutes ago (i.e. the job is due)
        jobs = load_jobs()
        old_next = (datetime.now() - timedelta(minutes=5)).isoformat()
        jobs[0]["next_run_at"] = old_next
        save_jobs(jobs)

        result = advance_next_run(job["id"])
        assert result is True

        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        new_next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next_dt > _hermes_now(), "next_run_at should be in the future after advance"

    def test_advances_cron_job(self, tmp_cron_dir):
        """Cron-expression jobs should have next_run_at bumped to the next occurrence."""
        pytest.importorskip("croniter")
        job = create_job(prompt="Daily wakeup", schedule="15 6 * * *")
        # Force next_run_at to 30 minutes ago
        jobs = load_jobs()
        old_next = (datetime.now() - timedelta(minutes=30)).isoformat()
        jobs[0]["next_run_at"] = old_next
        save_jobs(jobs)

        result = advance_next_run(job["id"])
        assert result is True

        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        new_next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next_dt > _hermes_now(), "next_run_at should be in the future after advance"

    def test_skips_oneshot_job(self, tmp_cron_dir):
        """One-shot jobs should NOT be advanced — they need to retry on restart."""
        job = create_job(prompt="Run once", schedule="30m")
        original_next = get_job(job["id"])["next_run_at"]

        result = advance_next_run(job["id"])
        assert result is False

        updated = get_job(job["id"])
        assert updated["next_run_at"] == original_next, "one-shot next_run_at should be unchanged"

    def test_nonexistent_job_returns_false(self, tmp_cron_dir):
        result = advance_next_run("nonexistent-id")
        assert result is False

    def test_already_future_stays_future(self, tmp_cron_dir):
        """If next_run_at is already in the future, advance keeps it in the future (no harm)."""
        job = create_job(prompt="Future job", schedule="every 1h")
        # next_run_at is already set to ~1h from now by create_job
        advance_next_run(job["id"])
        # Regardless of return value, the job should still be in the future
        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        new_next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next_dt > _hermes_now(), "next_run_at should remain in the future"

    def test_crash_safety_scenario(self, tmp_cron_dir):
        """Simulate the crash-loop scenario: after advance, the job should NOT be due."""
        job = create_job(prompt="Crash test", schedule="every 1h")
        # Force next_run_at to 5 minutes ago (job is due)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        # Job should be due before advance
        due_before = get_due_jobs()
        assert len(due_before) == 1

        # Advance (simulating what tick() does before run_job)
        advance_next_run(job["id"])

        # Now the job should NOT be due (simulates restart after crash)
        due_after = get_due_jobs()
        assert len(due_after) == 0, "Job should not be due after advance_next_run"


class TestConcurrentJobsJsonRMW:
    """Verify jobs_transaction serializes read-modify-write cycles.

    After scheduler.tick() was rewritten to dispatch jobs to a worker
    thread, the dispatcher keeps calling reserve_for_dispatch /
    advance_next_run while the worker calls mark_job_run. Without
    serialization, multiple actors can load jobs.json, each mutate a
    different job, and race their save — second save clobbers first.
    """

    def test_no_updates_lost_under_concurrent_in_process_rmw(self, tmp_cron_dir):
        """In-process: concurrent mark_job_run(A) + advance_next_run(B) keep both updates."""
        import threading
        from cron.jobs import create_job, mark_job_run, advance_next_run, load_jobs, save_jobs

        # Two recurring jobs, both stale so advance_next_run will mutate them.
        job_a = create_job(prompt="A", schedule="every 1h")
        job_b = create_job(prompt="B", schedule="every 1h")
        jobs = load_jobs()
        past = (datetime.now() - timedelta(minutes=30)).isoformat()
        for j in jobs:
            j["next_run_at"] = past
        save_jobs(jobs)

        # Fire many concurrent pairs. Without the lock, at least one run's
        # mutation would be lost ~always under contention.
        ITERATIONS = 40
        errors: list[str] = []

        def worker_mark(i):
            try:
                mark_job_run(job_a["id"], success=True, error=None)
            except Exception as e:
                errors.append(f"mark[{i}]: {e}")

        def worker_advance(i):
            try:
                advance_next_run(job_b["id"])
            except Exception as e:
                errors.append(f"advance[{i}]: {e}")

        threads: list[threading.Thread] = []
        for i in range(ITERATIONS):
            threads.append(threading.Thread(target=worker_mark, args=(i,)))
            threads.append(threading.Thread(target=worker_advance, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Worker errors: {errors[:5]}"

        # Final state: job A has been marked ok (last_run_at not None);
        # job B has next_run_at in the future. If the lock failed, one of
        # these would still be at its pre-concurrency state.
        final = load_jobs()
        final_a = next(j for j in final if j["id"] == job_a["id"])
        final_b = next(j for j in final if j["id"] == job_b["id"])

        assert final_a.get("last_run_at") is not None, "job A mark_job_run was lost to a race"
        assert final_a.get("last_status") == "ok"

        from cron.jobs import _ensure_aware, _hermes_now
        b_next_dt = _ensure_aware(datetime.fromisoformat(final_b["next_run_at"]))
        assert b_next_dt > _hermes_now(), "job B advance_next_run was lost to a race"

    def test_no_updates_lost_across_processes(self, tmp_path):
        """Cross-process: two subprocess-level actors running
        mark_job_run(A) + advance_next_run(B) must both survive the save.

        Regression for Mendi R1 Finding 2. Without a cross-process file
        lock, the in-process ``threading.Lock`` would not prevent
        process A's save from clobbering process B's mutation (or vice
        versa). Uses subprocess to force a second OS process sharing the
        same HERMES_HOME.
        """
        import os as _os
        import subprocess
        import sys
        import textwrap
        import cron.jobs as jobs_mod
        from cron.jobs import create_job, load_jobs, save_jobs, mark_job_run, _ensure_aware, _hermes_now

        hermes_home = tmp_path / "xproc_hermes"
        cron_dir = hermes_home / "cron"
        cron_dir.mkdir(parents=True)

        # Patch parent's cron.jobs constants to the shared HERMES_HOME so
        # the subprocess (which uses the HERMES_HOME env var) and the
        # parent both hit the same jobs.json + .jobs.lock.
        with patch.object(jobs_mod, "HERMES_DIR", hermes_home), \
             patch.object(jobs_mod, "CRON_DIR", cron_dir), \
             patch.object(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json"), \
             patch.object(jobs_mod, "OUTPUT_DIR", cron_dir / "output"), \
             patch.object(jobs_mod, "JOBS_LOCK_FILE", cron_dir / ".jobs.lock"):

            # Use recurring jobs so advance_next_run will actually mutate.
            job_a = create_job(prompt="cross-A", schedule="every 1h")
            job_b = create_job(prompt="cross-B", schedule="every 1h")
            jobs = load_jobs()
            past = (datetime.now() - timedelta(minutes=30)).isoformat()
            for j in jobs:
                j["next_run_at"] = past
            save_jobs(jobs)

            # Subprocess: advance_next_run(B) in a loop while parent
            # hammers mark_job_run(A) concurrently. Both share the same
            # jobs.json + jobs.lock via HERMES_HOME.
            subproc_script = textwrap.dedent(
                """
                import sys
                sys.path.insert(0, %(repo)r)
                from cron.jobs import advance_next_run
                for _ in range(200):
                    advance_next_run(%(jobid)r)
                print("SUBPROC_DONE")
                """
            ) % {
                "repo": str(jobs_mod.Path(jobs_mod.__file__).parent.parent),
                "jobid": job_b["id"],
            }

            env = {**_os.environ, "HERMES_HOME": str(hermes_home)}
            # Explicit PATH so subprocess Python can find fcntl / msvcrt
            # standard libs.
            proc = subprocess.Popen(
                [sys.executable, "-c", subproc_script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                for _ in range(200):
                    mark_job_run(job_a["id"], success=True, error=None)
                stdout, stderr = proc.communicate(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()

            assert proc.returncode == 0, (
                f"subprocess failed rc={proc.returncode} "
                f"stderr={stderr.decode()[:500]}"
            )
            assert b"SUBPROC_DONE" in stdout, (
                "subprocess did not reach completion marker"
            )

            final = load_jobs()
            final_a = next(j for j in final if j["id"] == job_a["id"])
            final_b = next(j for j in final if j["id"] == job_b["id"])

            assert final_a.get("last_run_at") is not None, (
                "job A mark_job_run was clobbered by a cross-process "
                "advance_next_run on B — the jobs file lock failed"
            )
            assert final_a.get("last_status") == "ok"

            b_next_dt = _ensure_aware(datetime.fromisoformat(final_b["next_run_at"]))
            assert b_next_dt > _hermes_now(), (
                "job B advance_next_run was clobbered by a cross-process "
                "mark_job_run on A — the jobs file lock failed"
            )


class TestMutationPathsWrappedInTransaction:
    """Regression for Mendi R2 Finding: create_job / update_job / remove_job
    must also hold jobs_transaction so their load-modify-save cycles don't
    race with concurrent mark_job_run / advance_next_run / reserve_for_dispatch.

    Mendi's deterministic repro: update_job loaded stale state, mark_job_run
    saved, update_job saved stale → final job lost last_run_at + last_status.
    This test reproduces the shape with concurrent threads and asserts both
    mutations survive.
    """

    def test_update_job_and_mark_job_run_dont_clobber(self, tmp_cron_dir):
        import threading
        from cron.jobs import create_job, update_job, mark_job_run, load_jobs

        job = create_job(prompt="R", schedule="every 1h")
        errors: list[str] = []

        def hammer_update():
            try:
                for _ in range(100):
                    update_job(job["id"], {"name": f"renamed-{_}"})
            except Exception as e:
                errors.append(f"update: {e}")

        def hammer_mark():
            try:
                for _ in range(100):
                    mark_job_run(job["id"], success=True, error=None)
            except Exception as e:
                errors.append(f"mark: {e}")

        t1 = threading.Thread(target=hammer_update)
        t2 = threading.Thread(target=hammer_mark)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Worker errors: {errors[:3]}"

        final = load_jobs()
        final_job = next(j for j in final if j["id"] == job["id"])
        # Both last_run_at (from mark_job_run) and a rename (from update_job)
        # must survive. If one path clobbered the other, one field would
        # still be at its pre-concurrency default.
        assert final_job.get("last_run_at") is not None, (
            "mark_job_run was clobbered by concurrent update_job"
        )
        assert final_job.get("last_status") == "ok"
        assert final_job.get("name", "").startswith("renamed-"), (
            "update_job was clobbered by concurrent mark_job_run"
        )

    def test_create_job_and_remove_job_are_atomic(self, tmp_cron_dir):
        """Concurrent create_job + remove_job don't leave the file in an
        inconsistent state (e.g. half-written additions, orphan entries)."""
        import threading
        from cron.jobs import create_job, remove_job, load_jobs

        created_ids: list[str] = []
        create_lock = threading.Lock()

        def hammer_create():
            for _ in range(30):
                j = create_job(prompt=f"c-{_}", schedule="every 1h")
                with create_lock:
                    created_ids.append(j["id"])

        def hammer_remove():
            # Remove some fraction of what gets created
            for _ in range(60):
                with create_lock:
                    target = created_ids[-1] if created_ids else None
                if target:
                    remove_job(target)

        t1 = threading.Thread(target=hammer_create)
        t2 = threading.Thread(target=hammer_remove)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # jobs.json must still load cleanly — no partial writes, no corrupt
        # state.  (If the lock failed, save_jobs's atomic rename still
        # prevents half-written files, but a stale-overwrite could leave
        # phantom "already deleted" entries lingering.)
        final = load_jobs()
        final_ids = {j["id"] for j in final}
        # Every remaining entry must be one that was created (no corrupt
        # orphan entries).
        assert final_ids.issubset(set(created_ids)), (
            "remaining entries are not a subset of created — state corrupted"
        )


class TestReserveForDispatch:
    """Tests for reserve_for_dispatch — unified dispatch-claim across kinds."""

    def test_recurring_advances_next_run_at(self, tmp_cron_dir):
        """Recurring jobs: reserve_for_dispatch behaves like advance_next_run."""
        from cron.jobs import create_job, reserve_for_dispatch, load_jobs, save_jobs, _ensure_aware, _hermes_now

        job = create_job(prompt="R", schedule="every 1h")
        jobs = load_jobs()
        past = (datetime.now() - timedelta(minutes=30)).isoformat()
        jobs[0]["next_run_at"] = past
        save_jobs(jobs)

        assert reserve_for_dispatch(job["id"]) is True
        updated = load_jobs()[0]
        new_next = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert new_next > _hermes_now(), "recurring next_run_at must advance forward"
        assert "in_flight_until" in updated, "recurring reservations still set the lease"

    def test_once_sets_dispatch_lease(self, tmp_cron_dir):
        """Once jobs: reserve_for_dispatch sets next_run_at and
        in_flight_until to a future lease, preventing re-dispatch by the
        next tick.
        """
        from cron.jobs import create_job, reserve_for_dispatch, load_jobs, save_jobs, _ensure_aware, _hermes_now

        job = create_job(prompt="O", schedule="30m")  # one-shot
        # Force it due.
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        assert reserve_for_dispatch(job["id"], stale_lease_seconds=1800) is True
        updated = load_jobs()[0]
        lease = _ensure_aware(datetime.fromisoformat(updated["in_flight_until"]))
        next_run = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        now = _hermes_now()
        assert lease > now, "lease must be in the future"
        assert next_run == lease, "once-job next_run_at should mirror the lease"

    def test_once_crash_recovery_after_lease(self, tmp_cron_dir):
        """If a once-job is reserved but the worker never marks it done,
        the job becomes due again after the lease expires.
        """
        from cron.jobs import create_job, reserve_for_dispatch, get_due_jobs, load_jobs, save_jobs

        job = create_job(prompt="O", schedule="30m")
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        # Reserve with a 1-second lease so we can observe expiry in-test.
        reserve_for_dispatch(job["id"], stale_lease_seconds=1)

        # Immediately: not due (lease is live).
        assert all(j["id"] != job["id"] for j in get_due_jobs())

        # After the lease expires, the job becomes due again.
        import time
        time.sleep(1.2)
        assert any(j["id"] == job["id"] for j in get_due_jobs()), (
            "expired lease must allow retry"
        )

    def test_mark_job_run_clears_lease(self, tmp_cron_dir):
        """mark_job_run() clears in_flight_until as part of finalize."""
        from cron.jobs import create_job, reserve_for_dispatch, mark_job_run, load_jobs, save_jobs

        job = create_job(prompt="R", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=30)).isoformat()
        save_jobs(jobs)

        reserve_for_dispatch(job["id"])
        assert "in_flight_until" in load_jobs()[0]

        mark_job_run(job["id"], success=True, error=None)
        assert "in_flight_until" not in load_jobs()[0]


class TestGetDueJobs:
    def test_past_due_within_window_returned(self, tmp_cron_dir):
        """Jobs within the dynamic grace window are still considered due (not stale).

        For an hourly job, grace = 30 min (half the period, clamped to [120s, 2h]).
        """
        job = create_job(prompt="Due now", schedule="every 1h")
        # Force next_run_at to 10 minutes ago (within the 30-min grace for hourly)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=10)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == job["id"]

    def test_stale_past_due_skipped(self, tmp_cron_dir):
        """Recurring jobs past their dynamic grace window are fast-forwarded, not fired.

        For an hourly job, grace = 30 min. Setting 35 min late exceeds the window.
        """
        job = create_job(prompt="Stale", schedule="every 1h")
        # Force next_run_at to 35 minutes ago (beyond the 30-min grace for hourly)
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=35)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 0
        # next_run_at should be fast-forwarded to the future
        updated = get_job(job["id"])
        from cron.jobs import _ensure_aware, _hermes_now
        next_dt = _ensure_aware(datetime.fromisoformat(updated["next_run_at"]))
        assert next_dt > _hermes_now()

    def test_future_not_returned(self, tmp_cron_dir):
        create_job(prompt="Not yet", schedule="every 1h")
        due = get_due_jobs()
        assert len(due) == 0

    def test_disabled_not_returned(self, tmp_cron_dir):
        job = create_job(prompt="Disabled", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["enabled"] = False
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 0

    def test_broken_recent_one_shot_without_next_run_is_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 22, 30, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        run_at = "2026-03-18T04:22:00+00:00"
        save_jobs(
            [{
                "id": "oneshot-recover",
                "name": "Recover me",
                "prompt": "Word of the day",
                "schedule": {"kind": "once", "run_at": run_at, "display": "once at 2026-03-18 04:22"},
                "schedule_display": "once at 2026-03-18 04:22",
                "repeat": {"times": 1, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-03-18T04:21:00+00:00",
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        due = get_due_jobs()

        assert [job["id"] for job in due] == ["oneshot-recover"]
        assert get_job("oneshot-recover")["next_run_at"] == run_at

    def test_broken_stale_one_shot_without_next_run_is_not_recovered(self, tmp_cron_dir, monkeypatch):
        now = datetime(2026, 3, 18, 4, 30, 0, tzinfo=timezone.utc)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        save_jobs(
            [{
                "id": "oneshot-stale",
                "name": "Too old",
                "prompt": "Word of the day",
                "schedule": {"kind": "once", "run_at": "2026-03-18T04:22:00+00:00", "display": "once at 2026-03-18 04:22"},
                "schedule_display": "once at 2026-03-18 04:22",
                "repeat": {"times": 1, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": "2026-03-18T04:21:00+00:00",
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "deliver": "local",
                "origin": None,
            }]
        )

        assert get_due_jobs() == []
        assert get_job("oneshot-stale")["next_run_at"] is None


class TestSaveJobOutput:
    def test_creates_output_file(self, tmp_cron_dir):
        output_file = save_job_output("test123", "# Results\nEverything ok.")
        assert output_file.exists()
        assert output_file.read_text() == "# Results\nEverything ok."
        assert "test123" in str(output_file)
