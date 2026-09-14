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
        self.assertEqual(
            tuple(inspect.signature(AirPianoTransformer).parameters),
            (
                "causal_frame_attention",
                "include_chord_duration",
                "zero_hand_input",
            ),
        )
        model = AirPianoTransformer().eval()
        hands = torch.randn(2, 3, 2, 21, 3)
        presence = torch.ones(2, 3, 2, dtype=torch.bool)
        chords = torch.zeros(2, 4, 328)
        chords[:, :, [39, 43, 46]] = 1.0
        chords[:, :, 128] = 1.0
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

    def test_duration_conditioning_uses_328_features_and_backpropagates(self) -> None:
        torch.manual_seed(5)
        model = AirPianoTransformer(include_chord_duration=True).train()
        hands = torch.randn(1, 2, 2, 21, 3)
        presence = torch.ones(1, 2, 2, dtype=torch.bool)
        chords = torch.zeros(1, 2, 328, requires_grad=True)
        with torch.no_grad():
            chords[0, 0, [60, 128 + 20]] = 1.0
            chords[0, 1, [64, 128 + 40]] = 1.0
        frame_mask = torch.ones(1, 2, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)

        logits = model(hands, presence, chords, frame_mask, chord_mask)
        logits.sum().backward()

        self.assertEqual(model.chord_embedding[0].in_features, 128)
        self.assertIsNotNone(model.chord_duration_embedding)
        assert model.chord_duration_embedding is not None
        self.assertEqual(model.chord_duration_embedding[0].in_features, 200)
        self.assertEqual(tuple(logits.shape), (1, 2, 2))
        self.assertGreater(float(chords.grad[..., 128:].abs().sum()), 0.0)

        with self.assertRaisesRegex(ValueError, "328"):
            model(hands, presence, chords[..., :128], frame_mask, chord_mask)

    def test_pitch_only_model_ignores_duration_suffix(self) -> None:
        torch.manual_seed(6)
        model = AirPianoTransformer().eval()
        hands = torch.randn(1, 2, 2, 21, 3)
        presence = torch.ones(1, 2, 2, dtype=torch.bool)
        chords = torch.zeros(1, 2, 328)
        chords[0, 0, [60, 128 + 10]] = 1.0
        chords[0, 1, [64, 128 + 20]] = 1.0
        altered = chords.clone()
        altered[..., 128:] = torch.randn_like(altered[..., 128:])
        frame_mask = torch.ones(1, 2, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)

        with torch.no_grad():
            expected = model(hands, presence, chords, frame_mask, chord_mask)
            actual = model(hands, presence, altered, frame_mask, chord_mask)

        self.assertIsNone(model.chord_duration_embedding)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_absent_coordinates_are_ignored_without_masking_the_frame(self) -> None:
        torch.manual_seed(7)
        model = AirPianoTransformer().eval()
        hands = torch.randn(1, 2, 2, 21, 3)
        presence = torch.tensor([[[True, False], [False, False]]])
        altered = hands.clone()
        altered[~presence] = torch.randn_like(altered[~presence]) * 10_000.0
        chords = torch.zeros(1, 2, 328)
        chords[0, 0, [60, 64, 67]] = 1.0
        chords[0, 1, [62, 65, 69]] = 1.0
        chords[:, :, 128] = 1.0
        frame_mask = torch.ones(1, 2, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)

        with torch.no_grad():
            expected = model(hands, presence, chords, frame_mask, chord_mask)
            actual = model(altered, presence, chords, frame_mask, chord_mask)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(bool((actual != 0.0).all()))

    def test_zero_hand_input_ignores_coordinates_and_presence(self) -> None:
        torch.manual_seed(8)
        model = AirPianoTransformer(zero_hand_input=True).eval()
        hands = torch.randn(1, 3, 2, 21, 3)
        presence = torch.ones(1, 3, 2, dtype=torch.bool)
        altered_hands = torch.randn_like(hands) * 10_000.0
        altered_presence = torch.zeros_like(presence)
        chords = torch.zeros(1, 2, 328)
        chords[0, 0, 60] = 1.0
        chords[0, 1, 64] = 1.0
        chords[:, :, 128] = 1.0
        frame_mask = torch.ones(1, 3, dtype=torch.bool)
        chord_mask = torch.ones(1, 2, dtype=torch.bool)

        with torch.no_grad():
            expected = model(
                hands, presence, chords, frame_mask, chord_mask
            )
            actual = model(
                altered_hands,
                altered_presence,
                chords,
                frame_mask,
                chord_mask,
            )

        self.assertTrue(model.zero_hand_input)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_causal_attention_prevents_future_frames_changing_prefix(self) -> None:
        torch.manual_seed(9)
        model = AirPianoTransformer(causal_frame_attention=True).eval()
        hands = torch.randn(1, 4, 2, 21, 3)
        presence = torch.ones(1, 4, 2, dtype=torch.bool)
        altered_hands = hands.clone()
        altered_hands[:, 2:] = torch.randn_like(altered_hands[:, 2:]) * 100.0
        altered_presence = presence.clone()
        altered_presence[:, 2:] = False
        chords = torch.zeros(1, 3, 328)
        chords[0, 0, 60] = 1.0
        chords[0, 1, 64] = 1.0
        chords[0, 2, 67] = 1.0
        chords[:, :, 128] = 1.0
        frame_mask = torch.ones(1, 4, dtype=torch.bool)
        chord_mask = torch.ones(1, 3, dtype=torch.bool)

        with torch.no_grad():
            expected = model(
                hands, presence, chords, frame_mask, chord_mask
            )
            actual = model(
                altered_hands,
                altered_presence,
                chords,
                frame_mask,
                chord_mask,
            )

        torch.testing.assert_close(
            actual[:, :2], expected[:, :2], rtol=0.0, atol=1e-6
        )

    def test_backward_reaches_hand_chord_transformer_and_scorer(self) -> None:
        torch.manual_seed(11)
        model = AirPianoTransformer().train()
        hands = torch.randn(1, 2, 2, 21, 3, requires_grad=True)
        presence = torch.ones(1, 2, 2, dtype=torch.bool)
        chords = torch.zeros(1, 2, 328, requires_grad=True)
        with torch.no_grad():
            chords[0, 0, [48, 52, 55]] = 1.0
            chords[0, 1, [50, 53, 57]] = 1.0
            chords[:, :, 128] = 1.0
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
