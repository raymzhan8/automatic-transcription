"""Model parameter count and forward-pass tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.framewise_models import (  # noqa: E402
    FramewiseBiGRUModel,
    FramewiseTCNModel,
    count_params,
)


def test_b0_param_count():
    model = FramewiseTCNModel(predict_pitch=False)
    assert count_params(model) == 434_500


def test_b1_param_count():
    model = FramewiseTCNModel(predict_pitch=True)
    assert count_params(model) == 434_629


def test_tcn_type_and_pitch_forward():
    model = FramewiseTCNModel(predict_pitch=True)
    x = torch.randn(2, 1, 360, 400)
    logits, pitch = model(x)
    assert logits.shape == (2, 400, 4)
    assert pitch.shape == (2, 400)


def test_c_shell_param_count():
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    assert count_params(model) == 305_221
    from training.train_framewise import expected_param_count

    assert expected_param_count({"task": "type_and_pitch", "architecture": "bigru"}) == 305_221


def test_tcn_type_only_forward():
    model = FramewiseTCNModel(predict_pitch=False)
    x = torch.randn(2, 1, 360, 400)
    out = model(x)
    assert out.shape == (2, 400, 4)


def test_bigru_forward_with_lengths():
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=False)
    x = torch.randn(2, 1, 360, 400)
    lengths = torch.tensor([400, 300])
    out = model(x, lengths)
    assert out.shape == (2, 400, 4)


def test_c_type_and_pitch_shapes():
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    x = torch.randn(2, 1, 360, 400)
    logits, pitch = model(x)
    assert logits.shape == (2, 400, 4)
    assert pitch.shape == (2, 400)


def test_bigru_all_short_batch_keeps_total_length():
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    model.eval()
    x = torch.randn(2, 1, 360, 400)
    with torch.no_grad():
        logits, pitch = model(x, torch.tensor([80, 140]))
    assert logits.shape == (2, 400, 4)
    assert pitch.shape == (2, 400)


def _gru_from_emb(model: FramewiseBiGRUModel, spec: torch.Tensor, lengths: torch.Tensor | None):
    emb = model.freq_cnn(spec)
    h_seq = emb.transpose(1, 2)
    t_full = h_seq.size(1)
    if lengths is not None:
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            h_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = model.gru(packed)
        h_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=t_full
        )
    else:
        h_seq, _ = model.gru(h_seq)
    return h_seq


def test_bigru_packed_prefix_matches_truncated():
    torch.manual_seed(0)
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    model.eval()
    t_full, length = 400, 120
    x = torch.randn(1, 1, 360, t_full)
    x[:, :, :, length:] = 0
    with torch.no_grad():
        logits_packed, pitch_packed = model(x, torch.tensor([length]))
        assert logits_packed.shape == (1, t_full, 4)
        emb = model.freq_cnn(x)
        h_packed = _gru_from_emb(model, x, torch.tensor([length]))
        h_trunc, _ = model.gru(emb.transpose(1, 2)[:, :length])
        torch.testing.assert_close(h_packed[:, :length], h_trunc, atol=1e-5, rtol=1e-4)
        logits_trunc = model.type_head(h_trunc)
        pitch_trunc = model.pitch_head(h_trunc).squeeze(-1)
        torch.testing.assert_close(logits_packed[:, :length], logits_trunc, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(pitch_packed[:, :length], pitch_trunc, atol=1e-5, rtol=1e-4)


def test_bigru_padding_does_not_leak_into_prefix():
    torch.manual_seed(0)
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    model.eval()
    t_full, length = 400, 120
    x = torch.randn(1, 1, 360, t_full)
    x[:, :, :, length:] = 0
    with torch.no_grad():
        h_packed = _gru_from_emb(model, x, torch.tensor([length]))
        h_unpacked = _gru_from_emb(model, x, None)
        emb = model.freq_cnn(x)
        h_trunc, _ = model.gru(emb.transpose(1, 2)[:, :length])
        torch.testing.assert_close(h_packed[:, :length], h_trunc, atol=1e-5, rtol=1e-4)
        leak = (h_unpacked[:, :length] - h_trunc).abs().max().item()
        assert leak > 1e-3


def test_bigru_mixed_lengths_item_matches_truncated():
    torch.manual_seed(1)
    model = FramewiseBiGRUModel(gru_layers=1, predict_pitch=True)
    model.eval()
    x = torch.randn(3, 1, 360, 400)
    lengths = torch.tensor([400, 250, 90])
    with torch.no_grad():
        logits, pitch = model(x, lengths)
        assert logits.shape == (3, 400, 4)
        assert pitch.shape == (3, 400)
        emb = model.freq_cnn(x)
        h_packed = _gru_from_emb(model, x, lengths)
        h_item, _ = model.gru(emb[1:2].transpose(1, 2)[:, :250])
        torch.testing.assert_close(h_packed[1:2, :250], h_item, atol=1e-5, rtol=1e-4)
