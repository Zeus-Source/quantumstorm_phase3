import os
import markdown
from xhtml2pdf import pisa


def convert_md_to_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(base_dir, "FINAL_5_PAGE_WRITEUP.md")
    pdf_path = os.path.join(base_dir, "FINAL_5_PAGE_WRITEUP.pdf")

    # Read markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(md_text, extensions=["extra", "tables"])

    # Wrap in HTML template with professional Times New Roman styling
    # Note: Use simpler CSS to ensure xhtml2pdf parses it correctly.
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: letter;
                margin-left: 1in;
                margin-right: 1in;
                margin-top: 1in;
                margin-bottom: 1in;
            }}
            body {{
                font-family: "Times New Roman", "Times", serif;
                font-size: 11pt;
                line-height: 1.3;
                color: #000000;
            }}
            h1, h2, h3, h4 {{
                font-family: "Times New Roman", "Times", serif;
                font-weight: bold;
                color: #000000;
            }}
            h1 {{
                font-size: 16pt;
                margin-top: 0;
                margin-bottom: 18pt;
                text-align: center;
                text-transform: uppercase;
            }}
            h2 {{
                font-size: 13pt;
                margin-top: 16pt;
                margin-bottom: 8pt;
                border-bottom: 1px solid #333333;
                padding-bottom: 3pt;
            }}
            h3 {{
                font-size: 11pt;
                margin-top: 12pt;
                margin-bottom: 6pt;
            }}
            p {{
                margin-top: 0;
                margin-bottom: 8pt;
                text-align: justify;
            }}
            ul, ol {{
                margin-top: 0;
                margin-bottom: 8pt;
                padding-left: 20pt;
            }}
            li {{
                margin-bottom: 4pt;
                text-align: justify;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10pt;
                margin-bottom: 10pt;
                font-size: 10pt;
            }}
            th, td {{
                border: 1px solid #000000;
                padding: 5pt;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            pre, code {{
                font-family: "Courier New", Courier, monospace;
                font-size: 9.5pt;
                background-color: #f8f8f8;
            }}
            pre {{
                padding: 6pt;
                border: 1px solid #cccccc;
                margin-top: 8pt;
                margin-bottom: 8pt;
            }}
            hr {{
                border: 0;
                border-top: 1px solid #999999;
                margin-top: 12pt;
                margin-bottom: 12pt;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Export to PDF
    with open(pdf_path, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(html_template, dest=result_file)

    if pisa_status.err:
        print("Error during PDF generation")
    else:
        print(f"Successfully generated PDF: {pdf_path}")


if __name__ == "__main__":
    convert_md_to_pdf()
