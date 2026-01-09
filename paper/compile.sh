#!/bin/bash
# Compile the MQR arXiv paper
# Run this script from the paper/ directory

echo "Compiling Möbius Quantum Ring paper..."

# Check if we're in the right directory
if [ ! -f "mqr_arxiv.tex" ]; then
    echo "Error: mqr_arxiv.tex not found. Please run from the paper/ directory."
    exit 1
fi

# First pass
pdflatex -interaction=nonstopmode mqr_arxiv.tex

# Second pass for references
pdflatex -interaction=nonstopmode mqr_arxiv.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc *.loa

echo ""
echo "Done! Output: mqr_arxiv.pdf"
echo "Pages: $(pdfinfo mqr_arxiv.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo 'unknown')"