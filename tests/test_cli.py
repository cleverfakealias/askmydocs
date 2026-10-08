import pytest

from askmydocs.cli import build_parser, main, ui_command


def test_ui_listens_on_localhost_only() -> None:
    command = ui_command()

    assert "--server.address=localhost" in command
    assert "--browser.gatherUsageStats=false" in command


def test_empty_question_exits_before_loading_models(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["ask", "   "])

    assert exit_info.value.code == 2
    assert "the question is empty" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["status"], ["ui"], ["ingest", "a.pdf"], ["ask", "why?"]])
def test_parser_accepts_every_command(argv: list[str]) -> None:
    assert build_parser().parse_args(argv).command == argv[0]
