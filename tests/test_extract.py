from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from diseval.extract import TOOL, extract_one


def fake_client(content, calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(content=content)
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def tool_block(payload):
    return SimpleNamespace(type="tool_use", input=payload)


def test_returns_validated_mentions_and_drops_blanks():
    client = fake_client([tool_block({"diseases": [" colon cancer ", "", "  "]})])
    assert extract_one(client, "m", "text") == ["colon cancer"]


def test_empty_list_is_allowed():
    assert extract_one(fake_client([tool_block({"diseases": []})]), "m", "text") == []


def test_wrong_shape_is_rejected():
    with pytest.raises(ValidationError):
        extract_one(fake_client([tool_block({"diseases": "asthma"})]), "m", "text")


def test_missing_tool_call_raises():
    with pytest.raises(ValueError):
        extract_one(fake_client([SimpleNamespace(type="text", text="hi")]), "m", "text")


def test_request_forces_the_tool():
    calls = []
    extract_one(fake_client([tool_block({"diseases": []})], calls), "some-model", "a sentence")
    req = calls[0]
    assert req["model"] == "some-model"
    assert req["tool_choice"] == {"type": "tool", "name": TOOL["name"]}
    assert "a sentence" in req["messages"][0]["content"]
