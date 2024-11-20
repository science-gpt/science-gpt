import pymupdf4llm
from pathlib import Path

def custom_hdr_info(span, page=None):
    text = span.get("text", "").strip()
    font_size = span.get("size", 0)
    flags = span.get("flags", 0)  # Bold/Italic detection
    is_bold = flags & 2
    is_italic = flags & 1

    # Header recognition: bolded and starting with a number
    if is_bold and text[0].isdigit():  # Text starts with a digit
        return "#" * text.count(".")  # Map to H1, H2, H3 based on numbering

    # Subheader logic: bolded text on a single line
    if is_bold and "\n" not in text:  # Single-line bolded text
        return "###"

    return ""  # Default: not a header

# Path to your PDF and output file
pdf_path = "/home/science-gpt/science-gpt/app/data/Tetraconazole PRD2012-29.pdf"
output_md_path = "output2.md"

# Convert the document to Markdown with refined header logic
md_text = pymupdf4llm.to_markdown(
    pdf_path,
    hdr_info=custom_hdr_info  # Use the custom header logic
)

# Save the Markdown text to a file
Path(output_md_path).write_bytes(md_text.encode())
print(f"Markdown saved to {output_md_path}")