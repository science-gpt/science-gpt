# convert the document to markdown
import pymupdf4llm
md_text = pymupdf4llm.to_markdown("/home/science-gpt/science-gpt/app/data/Tetraconazole PRD2012-29.pdf", table_strategy='lines')
# Write the text to some file in UTF8-encoding
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
