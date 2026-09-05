"""Focused contracts for the fixed AirPiano model and loss."""

from __future__ import annotations

import inspect
import math
import unittest

import torch
import torch.nn.functional as F

from airpiano.model.loss import masked_bce_loss
from airpiano.model.transformer import AirPianoTransformer


class AirPianoTransformerTest(unittest.TestCase):
    def test_fixed_api_shapes_padding_and_all_padded_samples(self) -> None:
        self.assertEqual(tuple(inspect.signature(AirPianoTransformer).parameters), ())
        model = AirPianoTransformer().eval()
        hands = torch.randn(2, 3, 2, 21, 3)
        presence = torch.ones(2, 3, 2, dtype=torch.bool)
        chords = torch.zeros(2, 4, 128)
        chords[:, :, [39, 43, 46]] = 1.0
        frame_mask = torch.tensor([[True, True, False], [False, False, False]])
        chord_mask = torch.tensor(
            [[True, True, False, False], [False, False, False, False]]
        )

        with torch.no_grad():
            logits = model(hands, presence, chords, frame_mask, chord_mask)

        self.assertEqual(tuple(logits.shape), (2, 3, 4))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        valid = frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)
        self.assertTrue(bool((logits[~valid] == 0.0).all()))
        self.assertTrue(bool((logits[valid] != 0.0).all()))

    def test_absent_coordinates_are_ignored_without_masking_the_frame(self) -> None:
        torch.manual_seed(7)
        model = AirPianoTransformer().eval()
        hands = torch.randn(1, 2, 2, 21, 3)
        presence = torch.tensor([[[True, False], [False, False]]])
        altered = hands.clone()
        altered[~presence] = torch.randn_like(altered[~presence]) * 10_000.0
        chords = torch.zeros(1, 2, 128)
        chords[0, 0, [60, 64, 67]] = 1.0
        chords[0, 1, [62, 65, 69]] = 1.0
        frame_mask = torch.ones(1, 2, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)

        with torch.no_grad():
            expected = model(hands, presence, chords, frame_mask, chord_mask)
            actual = model(altered, presence, chords, frame_mask, chord_mask)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(bool((actual != 0.0).all()))

    def test_backward_reaches_hand_chord_transformer_and_scorer(self) -> None:
        torch.manual_seed(11)
        model = AirPianoTransformer().train()
        hands = torch.randn(1, 2, 2, 21, 3, requires_grad=True)
        presence = torch.ones(1, 2, 2, dtype=torch.bool)
        chords = torch.zeros(1, 2, 128, requires_grad=True)
        with torch.no_grad():
            chords[0, 0, [48, 52, 55]] = 1.0
            chords[0, 1, [50, 53, 57]] = 1.0
        frame_mask = torch.ones(1, 2, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)
        target = torch.tensor([[[1.0, 0.0], [0.5, 0.0]]])

        logits = model(hands, presence, chords, frame_mask, chord_mask)
        loss = masked_bce_loss(logits, target, frame_mask, chord_mask)
        loss.backward()

        self.assertTrue(math.isfinite(float(loss.detach())))
        for gradient in (
            hands.grad,
            chords.grad,
            model.hand_embedding[0].weight.grad,
            model.chord_embedding[0].weight.grad,
            model.chord_encoder.layers[0].self_attn.in_proj_weight.grad,
            model.frame_layers[0].cross_attention.in_proj_weight.grad,
            model.scorer.frame_query.weight.grad,
        ):
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.0)


class MaskedBceLossTest(unittest.TestCase):
    def test_matches_bce_on_valid_cells_and_masks_gradients(self) -> None:
        logits = torch.tensor(
            [[[2.0, -1.0, 8.0], [0.5, 3.0, -8.0]]],
            requires_grad=True,
        )
        target = torch.tensor([[[1.0, 0.0, 0.0], [0.25, 1.0, 1.0]]])
        frame_mask = torch.tensor([[True, False]])
        chord_mask = torch.tensor([[True, True, False]])
        valid = frame_mask.unsqueeze(-1) & chord_mask.unsqueeze(1)

        actual = masked_bce_loss(logits, target, frame_mask, chord_mask)
        expected = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
        torch.testing.assert_close(actual, expected)

        actual.backward()
        self.assertGreater(float(logits.grad[valid].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[~valid].abs().sum()), 0.0)

    def test_rejects_a_batch_without_valid_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            masked_bce_loss(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, dtype=torch.bool),
                torch.ones(1, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
