"""serving/batch_gen.py — shard-split and record-parsing unit tests.

These exercise only the pure-Python logic (deterministic sharding, JSON
parsing of a completion into the driver's output record). batch_gen.py
imports vLLM lazily (inside a function, not at module scope), so this file
never touches vLLM/GPU and runs anywhere the rest of the suite does.
"""
from __future__ import annotations

import json

from serving.batch_gen import (
    build_output_record,
    prompt_sha256,
    select_shard,
    shard_line_indices,
)


def test_shard_line_indices_is_a_deterministic_partition():
    n_total, num_shards = 11, 4
    shards = [shard_line_indices(n_total, k, num_shards) for k in range(num_shards)]

    # Every index appears in exactly one shard.
    all_idxs = sorted(i for shard in shards for i in shard)
    assert all_idxs == list(range(n_total))

    # The split is the textbook i % num_shards rule.
    for k, shard in enumerate(shards):
        assert shard == [i for i in range(n_total) if i % num_shards == k]


def test_shard_line_indices_rejects_bad_shard_index():
    try:
        shard_line_indices(10, 4, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shard_index >= num_shards")


def test_select_shard_preserves_original_line_index_and_record():
    records = [{"persona_id": f"P{i:03d}", "prompt": f"prompt {i}", "attempt": 1} for i in range(9)]
    shard = select_shard(records, shard_index=2, num_shards=3)
    # shard 2 of 3 over 9 records -> indices 2, 5, 8
    assert [i for i, _ in shard] == [2, 5, 8]
    assert [r["persona_id"] for _, r in shard] == ["P002", "P005", "P008"]


def test_prompt_sha256_is_stable_and_content_sensitive():
    a = prompt_sha256("hello")
    b = prompt_sha256("hello")
    c = prompt_sha256("hello ")
    assert a == b
    assert a != c
    assert len(a) == 64  # hex sha256


def test_build_output_record_marks_gen_ok_on_clean_parseable_stop():
    payload = {"patterns": [], "rules": [], "voice": "v"}
    rec = build_output_record(
        persona_id="P001",
        prompt="some prompt",
        attempt=1,
        model="Qwen/Qwen3-8B",
        text=json.dumps(payload),
        finish_reason="stop",
    )
    assert rec["gen_ok"] is True
    assert rec["raw_json"] == payload
    assert rec["finish_reason"] == "stop"
    assert rec["persona_id"] == "P001"
    assert rec["attempt"] == 1
    assert rec["prompt_sha256"] == prompt_sha256("some prompt")


def test_build_output_record_truncated_output_is_not_gen_ok():
    payload = {"patterns": [], "rules": [], "voice": "v"}
    rec = build_output_record(
        persona_id="P002",
        prompt="p",
        attempt=1,
        model="m",
        text=json.dumps(payload),
        finish_reason="length",
    )
    # Parsed successfully but truncated -- surfaced, not silently dropped, and
    # not counted as a clean generation.
    assert rec["gen_ok"] is False
    assert rec["raw_json"] == payload
    assert rec["finish_reason"] == "length"


def test_build_output_record_unparseable_text_yields_null_raw_json():
    rec = build_output_record(
        persona_id="P003",
        prompt="p",
        attempt=1,
        model="m",
        text="not json at all {",
        finish_reason="stop",
    )
    assert rec["gen_ok"] is False
    assert rec["raw_json"] is None


def test_build_output_record_skipped_reason_bypasses_parsing():
    rec = build_output_record(
        persona_id="P004",
        prompt="p",
        attempt=4,
        model="m",
        text="",
        finish_reason=None,
        skipped_reason="attempt_budget_exceeded",
    )
    assert rec["gen_ok"] is False
    assert rec["raw_json"] is None
    assert rec["finish_reason"] == "attempt_budget_exceeded"
    assert rec["attempt"] == 4
