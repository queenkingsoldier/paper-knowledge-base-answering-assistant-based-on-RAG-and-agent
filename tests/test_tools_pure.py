import pytest

from agent.tools import build_registry
from agent.tools.calculator import safe_eval
from agent.tools.keywords import extract_keywords


def test_safe_eval_basic_arithmetic():
    assert safe_eval("3.14 * 2.56") == pytest.approx(8.0384)


def test_safe_eval_respects_precedence_and_parens():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20


def test_safe_eval_supports_power_and_unary():
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("-5 + 3") == -2


def test_safe_eval_rejects_function_calls():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_safe_eval_rejects_attribute_access():
    with pytest.raises(ValueError):
        safe_eval("(1).__class__")


def test_safe_eval_rejects_names():
    with pytest.raises(ValueError):
        safe_eval("open('x')")


def test_safe_eval_rejects_empty_and_syntax_error():
    with pytest.raises(ValueError):
        safe_eval("")
    with pytest.raises(ValueError):
        safe_eval("2 +")


def test_safe_eval_rejects_division_by_zero_with_clear_message():
    with pytest.raises(ValueError) as e:
        safe_eval("1 / 0")
    assert "零" in str(e.value) or "zero" in str(e.value).lower()


def test_extract_keywords_returns_top_n_distinct_terms():
    kws = extract_keywords("Transformer 注意力机制 Transformer 注意力机制 自监督", top_n=3)
    assert len(kws) <= 3 and len(set(kws)) == len(kws)


def test_extract_keywords_ranks_repeated_term_first():
    assert extract_keywords("扩散模型 扩散模型 扩散模型 图像生成", top_n=1) == ["扩散模型"] or \
           "扩散" in extract_keywords("扩散模型 扩散模型 扩散模型 图像生成", top_n=1)[0]


def test_extract_keywords_empty_text_returns_empty():
    assert extract_keywords("", top_n=5) == []


def test_extract_keywords_handles_pure_numbers():
    assert isinstance(extract_keywords("3.14 2.71 1.41", top_n=2), list)


def test_registry_has_the_three_pure_tools():
    reg = build_registry()
    for name in ("calculator", "current_time", "extract_keywords"):
        assert name in reg


def test_every_registered_tool_has_valid_json_schema():
    for name, tool in build_registry().items():
        assert tool.parameters["type"] == "object"
        assert isinstance(tool.parameters["properties"], dict)
        assert tool.description


def test_registry_tools_are_callable_and_return_str():
    reg = build_registry()
    assert isinstance(reg["calculator"].func(expression="1+1"), str)
    assert isinstance(reg["current_time"].func(), str)
    assert isinstance(reg["extract_keywords"].func(text="甲 乙 甲"), str)
