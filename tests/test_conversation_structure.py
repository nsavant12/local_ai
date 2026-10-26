"""Conversation example construction using invented messages only."""

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ConversationStructure import build_reply_examples, split_sessions


class ConversationStructureTests(unittest.TestCase):
    def test_session_boundary_keeps_exact_hour_but_splits_larger_gap(self):
        start = datetime(2026, 1, 1, 12)
        messages = [
            {"timestamp": start, "sender": "Ava", "message": "Lunch?"},
            {"timestamp": start + timedelta(hours=1), "sender": "Me", "message": "Yes"},
            {"timestamp": start + timedelta(hours=2, seconds=1), "sender": "Bo", "message": "Tea?"},
        ]
        self.assertEqual(split_sessions(messages), [messages[:2], messages[2:]])
        with self.assertRaises(ValueError):
            split_sessions(list(reversed(messages)))

    def setUp(self) -> None:
        self.messages = [
            {"sender": "Friend", "message": "wanna grab food?"},
            {"sender": "Me", "message": "yeah"},
            {"sender": "Me", "message": "give me ten"},
            {"sender": "Friend", "message": "meet there?"},
            {"sender": "Me", "message": "yep"},
        ]

    def test_groups_consecutive_messages_and_selects_own_replies(self) -> None:
        self.assertEqual(build_reply_examples(self.messages), [
            {
                "context": [{"sender": "Friend", "text": "wanna grab food?"}],
                "reply": "yeah\ngive me ten",
            },
            {
                "context": [
                    {"sender": "Friend", "text": "wanna grab food?"},
                    {"sender": "Me", "text": "yeah\ngive me ten"},
                    {"sender": "Friend", "text": "meet there?"},
                ],
                "reply": "yep",
            },
        ])

    def test_context_limit_keeps_only_preceding_turns(self) -> None:
        examples = build_reply_examples(self.messages, max_context_turns=1)
        self.assertEqual(examples[0]["context"], [
            {"sender": "Friend", "text": "wanna grab food?"},
        ])
        self.assertEqual(examples[1]["context"], [
            {"sender": "Friend", "text": "meet there?"},
        ])

    def test_skips_blank_messages_and_opening_own_turn(self) -> None:
        messages = [
            {"sender": "Self", "message": "hey"},
            {"sender": "Friend", "message": "  \n"},
            {"sender": "Friend", "message": "ready?"},
            {"sender": "Self", "message": " yep! "},
        ]
        examples = build_reply_examples(messages, self_sender="Self")
        self.assertEqual(examples, [{
            "context": [
                {"sender": "Self", "text": "hey"},
                {"sender": "Friend", "text": "ready?"},
            ],
            "reply": " yep! ",
        }])
        self.assertEqual(build_reply_examples([]), [])
        self.assertEqual(build_reply_examples([self.messages[0]]), [])

    def test_context_limit_must_be_positive(self) -> None:
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                build_reply_examples(self.messages, max_context_turns=limit)

    def test_does_not_mutate_input_or_share_context_dictionaries(self) -> None:
        original = deepcopy(self.messages)
        examples = build_reply_examples(self.messages)
        self.assertEqual(self.messages, original)
        examples[0]["context"][0]["text"] = "edited example"
        self.assertEqual(examples[1]["context"][0]["text"], "wanna grab food?")
        self.assertEqual(self.messages, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
