def is_significant(threshould_condition):
    def print_significant(
        significant="null hypothesis is rejected",
        not_significant="null hypothesis cannot be rejected"
    ):
        return lambda p: significant if threshould_condition(p) else not_significant
    return print_significant


def span_color_if(pred, below_color: str, uppercolor: str):
    """
    文字列をspanタグで囲んで返す.

    Usage
    -----
    span_when_significant = span_color_if(lambda p: p < 0.05, "blue", "red")
    span_when_significant(p_value)("Result of statistic test !")
    """
    return lambda p: lambda s: f"<span style='color: {below_color if pred(p) else uppercolor};'>{s}</span>"
