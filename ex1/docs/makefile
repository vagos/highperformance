.PHONY: clean watch

# PDFS=$(shell find . -name "*.tex" | sed s/.tex/.pdf/)
PDFS=build/main.pdf

all: $(PDFS)

build/%.pdf: %.tex
	mkdir -p build
	xelatex -output-directory=build $*.tex

clean:
	rm -r build

watch:
	find . -name "*.tex" | entr -r make
