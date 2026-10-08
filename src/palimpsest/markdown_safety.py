"""Make untrusted text safe to render as Markdown in the web UI.

Model replies and document text can carry prompt-injected Markdown. An embedded image
makes the browser fetch its URL as soon as the page renders, which can leak data to
that host. These helpers remove image embeds and make link targets visible.
"""

import re

_INLINE_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_REFERENCE_IMAGE = re.compile(r"!\[([^\]]*)\]\[[^\]]*\]")
_INLINE_LINK = re.compile(r"\[([^\]]+)\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def neutralize_markdown(text: str) -> str:
    """Remove image embeds and turn links into plain 'text (url)' pairs.

    Formatting such as bold text, lists, and code blocks is kept.

    Args:
        text: Markdown written by the chat model.

    Returns:
        Markdown that renders no remote content and hides no link targets.
    """
    text = _INLINE_IMAGE.sub(_removed_image, text)
    text = _REFERENCE_IMAGE.sub(_removed_image, text)
    return _INLINE_LINK.sub(r"\1 (\2)", text)


def escape_markdown(text: str) -> str:
    """Escape every Markdown control character, so the text renders literally.

    Args:
        text: Plain text from a document or a file name.

    Returns:
        Text with each Markdown control character preceded by a backslash.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _removed_image(match: re.Match[str]) -> str:
    alt_text = match.group(1).strip()
    return f"(image removed: {alt_text})" if alt_text else "(image removed)"
