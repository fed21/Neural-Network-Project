main:
	pdflatex main

full:
	pdflatex main && \
	bibtex main && \
	pdflatex main && \
	pdflatex main

cut: full
	pdftk main.pdf cat 1-11 output paper.pdf
	pdftk main.pdf cat 12-end output supp.pdf

clean:
	rm *.aux *.bbl *.blg *.log *.pdf *.dep *.brf *.fls *.out

run:
	evince main.pdf

compress:
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/default -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages -dCompressFonts=true -r150 -sOutputFile=suppl.pdf suppl_highRes.pdf

