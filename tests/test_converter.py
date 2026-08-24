from pathlib import Path

from tools.convert_txt_to_csv import convert, parse_line


def test_converter_accepts_expected_line_and_reports_skips(tmp_path: Path):
    line = "Timestamp: 1.2 ID: 0350 000 DLC: 2 05 28"
    row = parse_line(line)
    assert row[:3] == ["1.2", "0350", "2"]
    assert row[3:5] == ["05", "28"]
    assert len(row) == 11

    source = tmp_path / "input.txt"
    target = tmp_path / "nested" / "output.csv"
    source.write_text(line + "\nnot a CAN row\n", encoding="utf-8")
    stats = convert(source, target)
    assert stats == {"total_lines": 2, "written_rows": 1, "skipped_lines": 1}
    assert target.exists()
