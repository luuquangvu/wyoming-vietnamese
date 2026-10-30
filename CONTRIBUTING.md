# Contributing to Wyoming Vietnamese

Thanks for contributing to Wyoming Vietnamese, a local Vietnamese speech-to-text (STT) and text-to-speech (TTS) service for the Wyoming protocol. Please search existing [issues](https://github.com/luuquangvu/wyoming-vietnamese/issues) before starting substantial work.

## Development setup

Use a POSIX environment; Linux, WSL, or macOS are recommended. The project requires Python 3.14 or newer, [uv](https://docs.astral.sh/uv/), and Node.js 26 LTS.

```bash
git clone https://github.com/luuquangvu/wyoming-vietnamese.git
cd wyoming-vietnamese
uv sync --locked --all-groups
npm ci
```

Use `uv run --locked` for Python commands. Keep `uv.lock` synchronized with `pyproject.toml`, and keep `package-lock.json` synchronized with `package.json`.

Manage Python dependencies exclusively with uv; do not edit `uv.lock` manually.

## Making changes

- Start from an up-to-date `main` branch and keep each pull request focused.
- Read the affected code and tests before editing.
- Add regression tests for behavior changes.
- Update the Docker Compose files, Dockerfile, README, and other documentation when applicable.
- Keep configuration examples and documented defaults aligned with the code when changing environment variables, engines, or voices.
- Do not change the version in `pyproject.toml` in ordinary pull requests.
- Review the complete diff before requesting review.

## Tests and validation

Run a focused test while developing, for example:

```bash
uv run --locked pytest tests/test_stt.py
```

Run the full local gate before submitting:

```bash
uv run --locked tools/validate.py
```

Validation checks dependency alignment, Ruff, Ty, Pyright, Interrogate, Tally, Prettier, and the full pytest suite. It passes only when the output contains `VALIDATION_SUCCESS`; Ruff, Tally, and Prettier may modify files, so review the diff afterward.

When changing the Dockerfile, Compose files, model download logic, or bundled runtime assets, also verify that the image builds and starts locally:

```bash
docker compose up --build -d
```

Changes that affect hardware-specific paths, such as AVX/AVX-512 or ARM DotProd optimizations, should be tested on representative hardware when possible.

## AI-assisted contributions

AI coding agents may be used. The author remains primarily responsible for every submitted change, including code produced with AI assistance. The author must understand the change, verify it against the existing code and tests, review the full diff, check for security and compatibility issues, and run the required validation. Do not submit generated code that has not been reviewed and tested.

## Pull requests

Describe what changed, why it changed, which engines and hardware are affected, and how it was tested. Include relevant security, model download or caching, performance, or compatibility considerations.

CI runs Ruff, Ty, Pyright, Interrogate, Tally, Prettier, and the pytest suite. It also builds the Docker image for `linux/amd64` and `linux/arm64` when Docker-related files change.

## Reporting issues

Use the [bug report](https://github.com/luuquangvu/wyoming-vietnamese/issues/new?template=bug_report.yaml), [feature request](https://github.com/luuquangvu/wyoming-vietnamese/issues/new?template=feature_request.yaml), or [other issue](https://github.com/luuquangvu/wyoming-vietnamese/issues/new?template=other.yaml) template. Include the Wyoming Vietnamese version or image tag, Home Assistant version, installation method, host OS and architecture, configured engines and voices, exact input, expected result, actual result, and relevant logs. Be sure to remove any private recordings, transcripts, or household details before posting.

## License

By contributing, you agree that your contributions are distributed under the project's [MIT License](LICENSE).
