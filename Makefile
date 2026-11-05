LATEX ?= tectonic

.PHONY: all article dist clean

all: article

article:
	mkdir -p build
	$(LATEX) -o build article.tex


dist: 
	rm -rf dist
	mkdir -p dist/misis-article
	cp misisarticle.cls article.tex README.md LICENSE dist/misis-article/
	cd dist && zip -r misis-article.zip misis-article
	rm -rf dist/misis-article

clean:
	rm -rf build dist
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml *.xdv *.synctex.gz *.fls *.fdb_latexmk
	rm -f article.pdf
