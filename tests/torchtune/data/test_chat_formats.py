# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import assert_dialogue_equal, MESSAGE_SAMPLE, TOOL_MESSAGE_SAMPLE
from torchtune.data import ChatMLFormat, Llama2ChatFormat, Message, MistralChatFormat, ToolChatMLFormat


class TestLlama2ChatFormat:
    expected_dialogue = [
        Message(
            role="user",
            content="[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. "
            "Your goal is to complete the task as faithfully as you can. While performing "
            "the task think step-by-step and justify your steps.\n<</SYS>>\n\nPlease "
            "briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary: [/INST] ",
        ),
        Message(
            role="assistant",
            content="A father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.",
        ),
    ]

    def test_format(self):
        actual = Llama2ChatFormat.format(MESSAGE_SAMPLE)
        assert_dialogue_equal(actual, self.expected_dialogue)


class TestMistralChatFormat:
    expected_dialogue = [
        Message(
            role="user",
            content="[INST] Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary: [/INST] ",
        ),
        Message(
            role="assistant",
            content="A father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.",
        ),
    ]

    def test_format(self):
        no_system_sample = MESSAGE_SAMPLE[1:]
        actual = MistralChatFormat.format(no_system_sample)
        assert_dialogue_equal(actual, self.expected_dialogue)

    def test_format_with_system_prompt_raises(self):
        with pytest.raises(
            ValueError, match="System prompts are not supported in MistralChatFormat"
        ):
            _ = MistralChatFormat.format(MESSAGE_SAMPLE)


class TestChatMLFormat:
    expected_dialogue = [
        Message(
            role="system",
            content="<|im_start|>system\nYou are an AI assistant. User will you give you a task. "
            "Your goal is to complete the task as faithfully as you can. While performing "
            "the task think step-by-step and justify your steps.<|im_end|>\n",
        ),
        Message(
            role="user",
            content="<|im_start|>user\nPlease "
            "briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary:<|im_end|>\n",
        ),
        Message(
            role="assistant",
            content="<|im_start|>assistant\nA father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.<|im_end|>",
        ),
    ]

    def test_format(self):
        actual = ChatMLFormat.format(MESSAGE_SAMPLE)
        assert_dialogue_equal(actual, self.expected_dialogue)

class TestToolChatMLFormat:
    expected_dialogue = [
        Message(
            role="system",
            content='<|im_start|>system\nYou are a helpful assistant with access to the following functions. '
            'Use them if required\n<tools>[{"type": "function", "function": {"name": "track_calories", "description": '
            '"Track daily calorie intake", "parameters": {"type": "object", "properties": {"meal": {"type": "string", '
            '"description": "The meal for which calories are being tracked"}, "calories": {"type": "number", "description": '
            '"The number of calories consumed"}, "date": {"type": "string", "format": "date", "description": "The date for '
            'which calories are being tracked"}}, "required": ["meal", "calories", "date"]}}}]</tools><|im_end|>\n',
        ),
        Message(
            role="user",
            content="<|im_start|>user\nHi, I had a pizza for lunch today which was about 800 calories. "
            "Can you track this for me?<|im_end|>\n",
        ),
        Message(
            role="assistant",
            content='<|im_start|>assistant\n<tool_call>\n{"id": "bDJIuDPjc", "type": "function", "function": '
            '{"name": "track_calories", "arguments": "{\\"meal\\": \\"pizza\\", \\"calories\\": 800, \\"date\\": \\"2022-03-01\\"}"}}\n</tool_call><|im_end|>\n'
        ),
        Message(
            role="tool",
            content='<|im_start|>tool\n{\"status\": \"success\", \"message\": \"Calories for your pizza meal have '
            'been successfully tracked for the date 2022-03-01\"}<|im_end|>\n'
        ),
        Message(
            role="assistant",
            content='<|im_start|>assistant\nGreat! The calories for your pizza meal have been successfully tracked for today.<|im_end|>\n'
        ),  
    ]

    def test_format(self):
        actual = ToolChatMLFormat.format(TOOL_MESSAGE_SAMPLE)
        assert_dialogue_equal(actual, self.expected_dialogue)

