from __future__ import annotations

import pytest

from unified_gateway.app.state.event_tracker import EventTracker


@pytest.mark.asyncio
async def test_event_tracker_summary_and_http_stats():
    tracker = EventTracker(max_events=1000)
    await tracker.record(
        event_type="http.request.completed",
        level="INFO",
        message="ok request",
        request_id="r1",
        data={"method": "GET", "path": "/health", "status_code": 200, "latency_ms": 10.0},
    )
    await tracker.record(
        event_type="http.request.completed",
        level="WARNING",
        message="bad request",
        request_id="r2",
        data={"method": "POST", "path": "/v1/embeddings", "status_code": 502, "latency_ms": 100.0},
    )
    await tracker.record(
        event_type="router.dispatch.completed",
        level="INFO",
        message="router ok",
        request_id="r2",
        data={"ok": True},
    )

    summary = await tracker.summary(since_minutes=60)
    assert summary["events_total"] == 3
    assert summary["http"]["requests_total"] == 2
    assert summary["http"]["status_2xx"] == 1
    assert summary["http"]["status_5xx"] == 1
    assert summary["http"]["latency_avg_ms"] == 55.0

    by_path = await tracker.http_request_stats(group_by="path", since_minutes=60)
    groups = {row["group"]: row for row in by_path["items"]}
    assert "/health" in groups
    assert "/v1/embeddings" in groups
    assert groups["/health"]["requests_total"] == 1

    events = await tracker.events(since_minutes=60, event_type="router.dispatch.completed", limit=10)
    assert events["count"] == 1
    assert events["items"][0]["request_id"] == "r2"


@pytest.mark.asyncio
async def test_event_tracker_invalid_group_by():
    tracker = EventTracker(max_events=1000)
    with pytest.raises(ValueError):
        await tracker.http_request_stats(group_by="invalid", since_minutes=60)
