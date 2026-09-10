# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Declan Porter's personal blog/portfolio: a static site (`site/`) showcasing ML and embedded/graphics
projects (`projects/`), backed by two small FastAPI services (`runner/`, `tracker/`) that are all
composed together with Docker in `docker-compose.yml`. There is no build tooling for the site —
it's hand-written HTML/CSS/JS served directly by nginx.

## Architecture

Three services, composed via `docker-compose.yml`:

- **`site/`** — static nginx site. Every page (`project-*.html`, article pages) is standalone HTML
  with inline `<script>` blocks, no bundler/framework. `nginx.conf` proxies `/api/` to the runner
  service and `/tracker/` to the tracker service, so client JS calls those paths directly (see
  `const RUNNER = '/api'` in project pages). The article index is static HTML in `articles.html`
  (one `<li class="post-list-item">` per post, newest first). Add new posts there by hand; it is
  deliberately not JS-rendered so crawlers and agents can read it.
- **`runner/app.py`** — single FastAPI app that loads *every* trained project's weights at startup
  and exposes inference endpoints (`/run/parabola`, `/run/circle-classifier`, `/run/mnist`,
  `/run/diffusion-circle`, `/run/diffusion-mnist`, `/run/vae-colors`, plus matching `/data/*`
  endpoints for pre-computed datasets). This is what the interactive demos on the project pages
  call through the `/api/` proxy.
- **`tracker/app.py`** — tiny FastAPI + SQLite page-view counter (`POST /ping?page=...`,
  `GET /stats`). DB path via `DB_PATH` env var, defaults to `/data/tracker.db` (a docker volume).

### Project convention (`projects/projectN-name/`)

Each numbered project directory is a self-contained ML experiment, generally one Python file plus
a saved `model.pt`. The convention the runner depends on:

- The file must define a class named exactly `Model` (subclass of `nn.Module`).
- Training code lives behind `if __name__ == "__main__":` — `runner/app.py` imports the file
  without executing training, then loads `model.pt` into `Model()`.
- To (re)train a project's model, run its script directly, e.g.:
  ```
  python projects/project2-classifier/circle_classifier.py
  ```
  This trains and overwrites `model.pt` next to the script.
- Some projects export extra module-level helpers the runner uses for demos (e.g.
  `sample_reverse`, `decode_grid`, `decode_point`, `latent_scatter` — see `project7`-`project9`).
  If you add a new interactive project, follow this pattern and wire it into `runner/app.py`'s
  load-and-endpoint sections.
- Not every project directory is runner-backed — some (`project5-parabola-cpp`, `project13-mesh`,
  `project14-cppjson`, `projectX-geomesh`) are standalone C++ or offline scripts with their own
  build scripts (`build_and_run.bat`, `CMakeLists.txt`) and aren't loaded by the FastAPI runner.

## Running locally

```
docker-compose up --build
```
site on :80 (proxying to runner :8000 and tracker :8001 internally). `tracker_data` is a named
volume for the SQLite file.

To run a single service outside Docker (e.g. iterating on the runner):
```
pip install -r runner/requirements.txt
uvicorn app:app --reload --app-dir runner
```
Note `runner/Dockerfile` builds with the repo root as context specifically so it can `COPY
projects/` — running `uvicorn` locally still expects `projects/` to be resolvable relative to
`runner/app.py` (it uses `os.path.dirname(__file__)`), so run it from the repo root.

`projects/requirements.txt` pins CUDA-specific torch (`cu124`) for local training/experimentation;
`runner/requirements.txt` is the lighter set actually used to serve inference in the container.

## Writing blog content

There's a detailed AI-writing-pattern checklist at `.claude/writing-skill.md` used for editing
article prose (removing "AI-isms" — em dashes, hedge words, template phrasing, etc.). Apply it when
writing or editing anything in `site/*.html` article pages or `socialnetworks/*.md` launch posts.
