# Makefile

ENV_NAME = ligo

.PHONY: env html clean

env:
	# Create or update the conda env from environment.yml (don’t activate)
	conda env update -n $(ENV_NAME) -f environment.yml || \
	mamba env update -n $(ENV_NAME) -f environment.yml
	@echo
	@echo "To use it:  conda activate $(ENV_NAME)"

html:
	# Build local HTML site into _build/site
	myst build --html
	@echo
	@echo "Open _build/site/index.html in a browser (local preview)."

clean:
	# Remove build artifacts and generated media
	rm -rf _build/*
	rm -rf figures/*
	rm -rf audio/*

