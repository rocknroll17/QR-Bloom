# Contributing to QR-Bloom

Thanks for taking the time to look. Issues, ideas, and pull requests are all
welcome.

## Quick start

```bash
git clone https://github.com/rocknroll17/QR-Bloom.git
cd QR-Bloom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make gallery        # open http://localhost:8000
```

You can also run everything in Docker — see the README's Docker section.

## How to contribute

1. **Open an issue first** for anything non-trivial. Quick fixes and typos can
   skip this and go straight to a PR.
2. **Branch from `main`** with a descriptive name (`feat/...`, `fix/...`,
   `docs/...`, `chore/...`).
3. **Keep PRs focused.** Smaller, single-purpose PRs land faster than mixed
   bundles.
4. **Update docs.** If you change a workflow, CLI flag, or env var, update
   the relevant section in `README.md`. If you change tree generation,
   rendering, or grid sizing, check that existing conventions (cell format,
   camera presets, isotropic grid scaling) still hold across `qrbloom/`
   and `docs/`.

## Code style

- Python: 4-space indent, type hints where they help, no unused imports.
- Keep functions small and named after what they do.
- Comments explain *why*, not *what*. Don't restate the code.

## Testing your change

- **Tree generation / grid sizing** — run `make train` for a few epochs
  (e.g. `EPOCHS=3 EPOCH_SIZE=2000 make train`) and check the
  `runs_all/epoch_*.png` previews look sane.
- **Gallery / UI** — `make gallery` and exercise the change in the browser.
- **Docker** — `docker build --target serve -t qrbloom-dev .` and run it
  locally; this is what CI builds.

If you're adding a feature that needs new infrastructure (a new model
output, a new endpoint, etc.), include a short test plan in the PR
description.

## Reporting bugs

Open an issue with the bug-report template. The most useful things to
include are: what you ran, what you expected, what happened (logs / stack
trace), and your environment (OS, Python version, GPU if relevant, image
tag or commit).

## License

By contributing you agree that your contributions are released under the
project's [MIT License](LICENSE).
