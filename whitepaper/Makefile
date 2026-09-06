.PHONY: all clean

all: generalist.pdf

generalist.pdf: main.tex references.bib evidence/results.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
	cp build/main.pdf generalist.pdf

clean:
	rm -rf build
