from __future__ import annotations

from pathlib import Path


def _assert_best_checkpoint_reloaded(script: str) -> None:
    text = Path(script).read_text(encoding="utf-8")
    load_pos = text.index('torch.load(best_ckpt_path')
    eval_pos = text.index('test_pred = evaluate')
    assert load_pos < eval_pos
    assert 'model.load_state_dict(checkpoint["state_dict"])' in text
    assert "best_epoch" in text
    assert "checkpoint_sha256" in text


def test_mlp_uses_best_checkpoint_for_test_evaluation():
    _assert_best_checkpoint_reloaded("experiments/run_train_mlp.py")


def test_resmlp_uses_best_checkpoint_for_test_evaluation():
    _assert_best_checkpoint_reloaded("experiments/run_train_resmlp.py")
