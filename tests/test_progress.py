import io

from src.progress import ProgressBar


def test_progress_bar_renders_and_closes():
    stream = io.StringIO()
    progress = ProgressBar(10, "测试", stream=stream, enabled=True)
    progress.update(5, status="处理中")
    progress.close()

    output = stream.getvalue()
    assert "测试" in output
    assert "5/10" in output
    assert "50.00%" in output
    assert output.endswith("\n")


def test_progress_bar_can_be_disabled():
    stream = io.StringIO()
    progress = ProgressBar(10, "测试", stream=stream, enabled=False)
    progress.update(10)
    progress.close()

    assert stream.getvalue() == ""
