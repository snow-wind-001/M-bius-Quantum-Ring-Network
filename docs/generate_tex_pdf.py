#!/usr/bin/env python3
"""
Standalone script to generate report.tex and report.pdf from report.md
Handles Mermaid diagrams, Chinese text, and mathematical formulas.
"""

import os
import re
import subprocess
import shutil
from pathlib import Path


def sanitize_mermaid_code(mermaid_code: str) -> str:
    """Sanitize Mermaid code for better compatibility.
    
    Mermaid has strict parsing rules:
    - Parentheses () inside node labels cause parsing errors
    - <br/> HTML tags need to be removed or handled
    - Special Unicode characters are not supported
    - Semicolons at line end can cause issues
    """
    
    # Step 1: Remove/replace HTML tags FIRST
    result = mermaid_code
    result = re.sub(r'<br\s*/?>', ' ', result)  # Remove all <br> variants
    
    # Step 2: Character replacement map
    replacements = {
        # Superscripts
        '⁻¹': '^-1', '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3',
        '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '⁺': '+', '⁻': '-', '⁽': '', '⁾': '',
        # Subscripts
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        # Greek letters (use short forms for Mermaid)
        'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'θ': 'th',
        'λ': 'L', 'μ': 'u', 'π': 'pi', 'σ': 's', 'τ': 't', 'φ': 'f', 'ω': 'w',
        'Δ': 'D', 'Σ': 'S', 'Ω': 'O', 'Φ': 'F',
        # Mathematical operators
        '⊙': '*', '⊗': 'x', '⊕': '+', '⊖': '-',
        '∑': 'Sum', '∏': 'Prod', '∫': 'Int',
        '≈': '~', '≠': '!=', '≤': '<=', '≥': '>=', '∞': 'inf', '∂': 'd',
        # Script letters
        '𝒥': 'J', 'ℒ': 'L', 'ℋ': 'H', '𝒯': 'T', '𝒪': 'O',
        '𝔼': 'E', 'ℝ': 'R', 'ℂ': 'C', 'ℤ': 'Z', 'ℕ': 'N',
        # Arrows - DON'T replace these globally as they might be inside labels!
        # '→': '-->', '←': '<--',  # REMOVED - breaks labels
        # Other
        '·': '.', '×': 'x', '÷': '/',
        '"': "'", '"': "'", ''': "'", ''': "'",
        'ᵀ': 'T', 'ᴴ': 'H', '†': 'dag',
        'ₓ': 'x', 'ₜ': 't', 'ₙ': 'n', 'ₘ': 'm',
        # Full-width punctuation to ASCII  
        '（': ' ', '）': ' ',  # Replace with space, not parentheses!
        '：': ' ', '；': ' ', '，': ' ',
        '。': '.', '！': '!', '？': '?',
        # Note: DON'T replace | as it's used for link labels in Mermaid
    }
    
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    # Step 3: Process lines
    lines = result.split('\n')
    processed_lines = []
    subgraph_counter = 0
    
    for line in lines:
        # Remove trailing semicolons (they can cause issues)
        line = re.sub(r';\s*$', '', line)
        
        # Handle subgraph with quoted Chinese title
        subgraph_match = re.match(r'^(\s*)subgraph\s+"([^"]+)"(.*)$', line)
        if subgraph_match:
            indent = subgraph_match.group(1)
            title = subgraph_match.group(2)
            rest = subgraph_match.group(3).strip()
            # Clean up the title - remove parentheses content that causes issues
            title = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', title).strip()
            subgraph_counter += 1
            # Remove any extra content after the subgraph definition
            line = f'{indent}subgraph sg{subgraph_counter}["{title}"]'
        
        # Handle subgraph with unquoted Chinese title  
        subgraph_match2 = re.match(r'^(\s*)subgraph\s+([^\[\s"]+[\u4e00-\u9fff][^\[\s]*)(.*)$', line)
        if subgraph_match2 and 'subgraph sg' not in line:
            indent = subgraph_match2.group(1)
            title = subgraph_match2.group(2)
            title = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', title).strip()
            subgraph_counter += 1
            line = f'{indent}subgraph sg{subgraph_counter}["{title}"]'
        
        processed_lines.append(line)
    
    result = '\n'.join(processed_lines)
    
    # Step 4: Fix node labels - the key is to handle parentheses INSIDE labels
    def fix_node_label(match):
        node_id = match.group(1)
        bopen = match.group(2)
        label = match.group(3)
        bclose = match.group(4)
        
        # Remove parentheses and their content from labels - they break Mermaid!
        # E.g., "J(x)" -> "J x" or "Cayley变换 Q = (I+S)(I-S)^-1" -> "Cayley变换 Q"
        label = re.sub(r'\([^)]*\)', '', label)  # Remove (...) content
        label = re.sub(r'\[[^\]]*\]', '', label)  # Remove [...] content inside labels
        # Also remove any remaining lone parentheses/brackets
        label = label.replace('(', '').replace(')', '')
        label = label.replace('[', '').replace(']', '')
        
        # Remove pipe characters used for absolute value/norm: |x| -> x, ||x|| -> x
        label = re.sub(r'\|\|([^|]+)\|\|', r'\1', label)  # ||...|| -> ...
        label = re.sub(r'\|([^|]+)\|', r'\1', label)  # |...| -> ...
        label = label.replace('|', '')  # Remove any remaining pipes
        
        # Also clean up double spaces
        label = re.sub(r'\s+', ' ', label).strip()
        
        # Remove problematic Unicode (keep Chinese)
        cleaned = ''.join(c for c in label if ord(c) <= 127 or '\u4e00' <= c <= '\u9fff')
        
        # Remove equals signs and other problematic chars that require complex quoting
        cleaned = cleaned.replace('=', ' ').replace('<', ' ').replace('>', ' ')
        cleaned = cleaned.replace(':', ' ').replace(';', ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return f'{node_id}{bopen}{cleaned}{bclose}'
    
    # Fix double brackets syntax error: A[text]] -> A[text]
    result = re.sub(r'(\[[^\]]+)\]\]', r'\1]', result)
    
    # Match node definitions: ID followed by bracket
    result = re.sub(
        r'\b([A-Za-z_][A-Za-z0-9_]*)(\[)([^\]]+)(\])',
        fix_node_label,
        result
    )
    
    # Also handle curly braces for decision nodes
    def fix_decision_label(match):
        node_id = match.group(1)
        label = match.group(2)
        label = re.sub(r'\([^)]*\)', '', label)
        label = re.sub(r'\s+', ' ', label).strip()
        cleaned = ''.join(c for c in label if ord(c) <= 127 or '\u4e00' <= c <= '\u9fff')
        return f'{node_id}{{{cleaned}}}'
    
    result = re.sub(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\{([^}]+)\}',
        fix_decision_label,
        result
    )
    
    # Handle double parentheses for circle nodes
    def fix_circle_label(match):
        node_id = match.group(1)
        label = match.group(2)
        label = re.sub(r'\([^)]*\)', '', label)
        label = re.sub(r'\s+', ' ', label).strip()
        cleaned = ''.join(c for c in label if ord(c) <= 127 or '\u4e00' <= c <= '\u9fff')
        return f'{node_id}(({cleaned}))'
    
    result = re.sub(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\(\(([^)]+)\)\)',
        fix_circle_label,
        result
    )
    
    return result


def render_mermaid_diagrams(md_path: Path) -> Path:
    """Render Mermaid diagrams and return path to processed markdown."""
    
    mmdc = shutil.which("mmdc")
    if not mmdc:
        print("WARNING: mmdc not installed, skipping Mermaid rendering")
        print("Install with: npm install -g @mermaid-js/mermaid-cli")
        return md_path
    
    content = md_path.read_text(encoding='utf-8')
    
    # Find mermaid blocks
    mermaid_pattern = r'```mermaid\n(.*?)```'
    matches = list(re.finditer(mermaid_pattern, content, re.DOTALL))
    
    if not matches:
        print("No Mermaid diagrams found")
        return md_path
    
    print(f"Found {len(matches)} Mermaid diagrams to render")
    
    # Create figures directory
    figures_dir = md_path.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    new_content = content
    rendered_count = 0
    
    for i, match in enumerate(matches, 1):
        mermaid_code = match.group(1).strip()
        
        # Sanitize
        sanitized = sanitize_mermaid_code(mermaid_code)
        
        # Write to file
        mmd_file = figures_dir / f"diagram_{i}.mmd"
        mmd_file.write_text(sanitized, encoding='utf-8')
        
        # Render
        png_file = figures_dir / f"diagram_{i}.png"
        cmd = [mmdc, "-i", str(mmd_file), "-o", str(png_file), "-b", "white", "-w", "1200", "-s", "2"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if png_file.exists():
                # Replace mermaid block with image
                img_md = f"![图 {i}](figures/diagram_{i}.png)"
                new_content = new_content.replace(match.group(0), img_md)
                rendered_count += 1
                print(f"  Diagram {i}: OK")
            else:
                error = result.stderr[:200] if result.stderr else "Unknown error"
                print(f"  Diagram {i}: FAILED - {error}")
        except Exception as e:
            print(f"  Diagram {i}: ERROR - {e}")
    
    print(f"Rendered {rendered_count}/{len(matches)} diagrams")
    
    # Save rendered markdown
    rendered_path = md_path.parent / "report_rendered.md"
    rendered_path.write_text(new_content, encoding='utf-8')
    
    return rendered_path


def fix_tex_content(tex_content: str) -> str:
    """Fix common issues in TeX content."""
    
    # Add necessary packages if missing
    packages = []
    if "\\usepackage{xeCJK}" not in tex_content and "\\usepackage{ctex}" not in tex_content:
        packages.append("\\usepackage{xeCJK}")
        packages.append("\\setCJKmainfont{Noto Serif CJK SC}")
    if "\\usepackage{amsmath}" not in tex_content:
        packages.append("\\usepackage{amsmath}")
    if "\\usepackage{amssymb}" not in tex_content:
        packages.append("\\usepackage{amssymb}")
    if "\\usepackage{graphicx}" not in tex_content:
        packages.append("\\usepackage{graphicx}")
    if "\\usepackage{longtable}" not in tex_content:
        packages.append("\\usepackage{longtable}")
    if "\\usepackage{booktabs}" not in tex_content:
        packages.append("\\usepackage{booktabs}")
    if "\\usepackage{float}" not in tex_content:
        packages.append("\\usepackage{float}")
    
    if packages:
        packages_str = "\n".join(packages)
        # Use string replacement instead of regex to avoid escaping issues
        tex_content = tex_content.replace(
            r'\begin{document}',
            f'{packages_str}\n\n\\begin{{document}}',
            1
        )
    
    # Fix Unicode math symbols
    math_replacements = {
        '⁻¹': '^{-1}', '⊙': '\\odot', '⊗': '\\otimes', '⊕': '\\oplus',
        '∑': '\\sum', '∏': '\\prod', '∫': '\\int', '∂': '\\partial',
        '∞': '\\infty', '≈': '\\approx', '≠': '\\neq', '≤': '\\leq', '≥': '\\geq',
        '→': '\\rightarrow', '←': '\\leftarrow', '↔': '\\leftrightarrow',
        '⇒': '\\Rightarrow', '⇐': '\\Leftarrow',
        'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta',
        'ε': '\\epsilon', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
        'π': '\\pi', 'σ': '\\sigma', 'τ': '\\tau', 'φ': '\\phi', 'ω': '\\omega',
        'Δ': '\\Delta', 'Σ': '\\Sigma', 'Φ': '\\Phi', 'Ω': '\\Omega',
        '𝒥': '\\mathcal{J}', 'ℒ': '\\mathcal{L}', 'ℋ': '\\mathcal{H}',
        '𝔼': '\\mathbb{E}', 'ℝ': '\\mathbb{R}', 'ℂ': '\\mathbb{C}',
        'ℤ': '\\mathbb{Z}', 'ℕ': '\\mathbb{N}',
        'ᵀ': '^{\\mathsf{T}}', '†': '^{\\dagger}',
    }
    
    for old, new in math_replacements.items():
        tex_content = tex_content.replace(old, new)
    
    # Fix \mathbf outside math mode - replace with \textbf
    tex_content = re.sub(r'(?<![\$\\])\\mathbf\{([^}]+)\}', r'\\textbf{\1}', tex_content)
    
    # Fix \mathcal outside math mode - just use the letter
    tex_content = re.sub(r'(?<![\$\\])\\mathcal\{([^}]+)\}', r'\1', tex_content)
    
    # Fix \mathbb outside math mode
    tex_content = re.sub(r'(?<![\$\\])\\mathbb\{([^}]+)\}', r'\1', tex_content)
    
    return tex_content


def generate_tex(md_path: Path) -> Path:
    """Generate TeX from Markdown using pandoc."""
    
    pandoc = shutil.which("pandoc")
    if not pandoc:
        print("ERROR: pandoc not installed")
        print("Install with: sudo apt-get install pandoc")
        return None
    
    tex_path = md_path.parent / "report.tex"
    
    cmd = [
        pandoc,
        str(md_path),
        "-o", str(tex_path),
        "--standalone",
        "-V", "documentclass=article",
        "-V", "fontsize=12pt",
        "-V", "geometry:margin=2.5cm",
        "-V", "CJKmainfont=Noto Serif CJK SC",
        "-V", "mainfont=DejaVu Serif",
        "-V", "sansfont=DejaVu Sans",
        "-V", "monofont=DejaVu Sans Mono",
        "--toc",
        "--toc-depth=3",
        "-N",
        "--from", "markdown+tex_math_dollars+pipe_tables",
        "--pdf-engine=xelatex",
    ]
    
    print("Generating report.tex...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(md_path.parent))
    
    if tex_path.exists():
        # Fix content
        tex_content = tex_path.read_text(encoding='utf-8')
        tex_content = fix_tex_content(tex_content)
        tex_path.write_text(tex_content, encoding='utf-8')
        print(f"TeX generated: {tex_path}")
        return tex_path
    else:
        print(f"ERROR: {result.stderr[:500]}")
        return None


def compile_tex(tex_path: Path) -> Path:
    """Compile TeX to PDF using XeLaTeX."""
    
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("ERROR: xelatex not installed")
        print("Install with: sudo apt install texlive-xetex texlive-fonts-recommended texlive-latex-extra texlive-lang-chinese fonts-noto-cjk")
        return None
    
    pdf_path = tex_path.with_suffix(".pdf")
    
    print("Compiling with XeLaTeX (2 passes)...")
    
    for run in range(2):
        cmd = [
            xelatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            tex_path.name,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(tex_path.parent),
            timeout=300,
        )
        
        if result.returncode != 0 and run == 1:
            log_file = tex_path.with_suffix(".log")
            if log_file.exists():
                log_content = log_file.read_text(encoding='utf-8', errors='ignore')
                errors = [l for l in log_content.split('\n') if l.startswith('!')]
                if errors:
                    print(f"TeX Errors: {errors[0][:100]}")
    
    if pdf_path.exists():
        print(f"PDF generated: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        
        # Cleanup aux files
        for ext in ['.aux', '.log', '.toc', '.out', '.fdb_latexmk', '.fls']:
            aux = tex_path.with_suffix(ext)
            if aux.exists():
                aux.unlink()
        
        return pdf_path
    else:
        print("PDF compilation failed. Check .log file for details.")
        return None


def generate_docx(md_path: Path) -> Path:
    """Generate DOCX from Markdown."""
    
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return None
    
    docx_path = md_path.parent / "report.docx"
    
    cmd = [
        pandoc,
        str(md_path),
        "-o", str(docx_path),
        "--from", "markdown+tex_math_dollars+pipe_tables",
    ]
    
    print("Generating report.docx...")
    subprocess.run(cmd, capture_output=True, cwd=str(md_path.parent))
    
    if docx_path.exists():
        print(f"DOCX generated: {docx_path}")
        return docx_path
    return None


def main():
    print("=" * 50)
    print("Document Generation Pipeline")
    print("=" * 50)
    
    # Find source file
    docs_dir = Path(__file__).parent
    md_path = docs_dir / "report.md"
    
    if not md_path.exists():
        print(f"ERROR: {md_path} not found")
        return
    
    print(f"Source: {md_path}\n")
    
    # Step 1: Render Mermaid diagrams
    print("[Step 1] Rendering Mermaid diagrams...")
    rendered_md = render_mermaid_diagrams(md_path)
    print()
    
    # Step 2: Generate DOCX
    print("[Step 2] Generating DOCX...")
    generate_docx(rendered_md)
    print()
    
    # Step 3: Generate TeX
    print("[Step 3] Generating TeX...")
    tex_path = generate_tex(rendered_md)
    if not tex_path:
        print("Failed to generate TeX")
        return
    print()
    
    # Step 4: Compile PDF
    print("[Step 4] Compiling PDF...")
    pdf_path = compile_tex(tex_path)
    print()
    
    print("=" * 50)
    if pdf_path:
        print("SUCCESS! Files generated:")
        print(f"  - {docs_dir / 'report.tex'}")
        print(f"  - {pdf_path}")
        print(f"  - {docs_dir / 'report.docx'}")
    else:
        print("Partial success. Check logs for errors.")
    print("=" * 50)


if __name__ == "__main__":
    main()
