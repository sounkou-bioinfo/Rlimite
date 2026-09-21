PKGNAME := $(shell sed -n 's/Package: *\([^ ]*\)/\1/p' DESCRIPTION)
PKGVERS := $(shell sed -n 's/Version: *\([^ ]*\)/\1/p' DESCRIPTION)

all: check

help:
	@printf '%s\n' \
	  'Common development targets:' \
	  '  make document    regenerate roxygen documentation' \
	  '  make readme      evaluate README.Rmd and render README.md' \
	  '  make install     install the current source tree' \
	  '  make test        run tinytest package tests' \
	  '  make build       build the source package' \
	  '  make check       run R CMD check --no-manual' \
	  '  make site        build the pkgdown site' \
	  '  make clean       remove build artifacts'

document:
	Rscript -e 'roxygen2::roxygenise()'

readme:
	Rscript -e 'rmarkdown::render("README.Rmd", output_format = "github_document")'
	Rscript tools/normalize-markdown.R README.md

install:
	R CMD INSTALL --preclean .

test: install
	Rscript -e 'tinytest::test_package("$(PKGNAME)")'

build:
	R CMD build .

check: build
	R CMD check --no-manual $(PKGNAME)_$(PKGVERS).tar.gz

site:
	Rscript -e 'pkgdown::build_site()'

clean:
	rm -rf $(PKGNAME)_$(PKGVERS).tar.gz $(PKGNAME).Rcheck

.PHONY: all help document readme install test build check site clean
