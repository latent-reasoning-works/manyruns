# Convenience targets. `make help` lists them.
.PHONY: help install install-public demo test lint

help:
	@echo "install       - install committed HEAD as a tool; excludes uncommitted changes"
	@echo "install-public - repair from the public repository; no GitHub login needed"
	@echo "demo          - manylatents run on swissroll with the embed recipe"
	@echo "test / lint   - pytest / ruff in the checkout (sync all used extras first)"

install:
	uv tool install --python 3.12 --force --reinstall-package manyruns "git+file://$(CURDIR)"

install-public:
	uv tool install --python 3.12 --force git+https://github.com/latent-reasoning-works/manyruns

demo:
	co-science run swissroll --recipe embed --engine manylatents --project demo

test:
	uv run pytest tests/ -q

lint:
	uv run ruff check manyruns tests
