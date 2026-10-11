from genai_platform.ingestion.loaders import html_to_text


def test_html_strips_script_and_keeps_headings():
    html = "<html><head><title>Doc</title></head><body><script>evil()</script><h1>Title</h1><p>Body text.</p></body></html>"
    text, title = html_to_text(html)
    assert "evil()" not in text
    assert "# Title" in text
    assert "Body text." in text
    assert title == "Doc"
