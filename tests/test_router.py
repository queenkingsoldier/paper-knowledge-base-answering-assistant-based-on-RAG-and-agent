from agent.router import classify


def test_pure_arithmetic_routes_to_calculator():
    d = classify("3.14 * 2.56 等于多少")
    assert d.kind == "direct_tool" and d.tool == "calculator"


def test_bare_number_is_not_treated_as_arithmetic():
    d = classify("3.14")
    assert d.kind == "agent"          # 只有数字，不是算式


def test_time_question_routes_to_current_time():
    for q in ("现在几点了", "今天日期是多少", "当前时间"):
        d = classify(q)
        assert d.kind == "direct_tool" and d.tool == "current_time"


def test_comparison_question_routes_to_agent_or_compare():
    d = classify("对比论文 A 和论文 B 的方法区别")
    assert d.kind == "agent" or d.tool == "paper_compare"


def test_normal_research_question_goes_to_agent():
    d = classify("ViT 用了什么数据集预训练")
    assert d.kind == "agent"


def test_empty_question_goes_to_agent_without_crash():
    assert classify("").kind == "agent"


def test_fullwidth_punctuation_question_goes_to_agent():
    assert classify("你好。世界！") .kind == "agent"


def test_decision_carries_human_readable_reason():
    assert classify("2+2").reason
