# ai-ml-small-projects

A collection of small, self-contained AI/ML projects. The main idea is to teach each segment of big infrastructure by building smaller clones of large production systems. Each folder is an independent project with its own setup, dependencies, and documentation — pick what you need and run with it.

## Projects

| Project | Description | Stack |
|---|---|---|
| [vllm-serve](./vllm-serve/) | Basic production-ready LLM serving with vLLM and FastAPI | vLLM, FastAPI, Docker |

## How to use

Each project is standalone. Navigate into any folder and follow its README:

```bash
cd vllm-serve
cat README.md
```

Projects use [uv](https://docs.astral.sh/uv/) for dependency management and [ruff](https://docs.astral.sh/ruff/) for linting.

## Contributing

To add a new project, create a folder with at minimum:

- `pyproject.toml` — dependencies and project config
- `README.md` — what it does, why, and how to run it
- `.env.example` — any required environment variables
