# Project Overview

This repository contains a collection of Python data-analysis scripts and a standalone HTML quiz-generator application.

## What's in the repo

- **`soat.txt` / `index.html`** — A single-file HTML application: a quiz generator for Physical Education ("Ứng dụng tạo trắc nghiệm - Môn Giáo dục thể chất"). Uses Tailwind CSS (via CDN), FileSaver.js, and html-docx-js. Client-side only, no backend.
- **`88.05.02.py`, `88.05.02-opt.py`, `chuongtrinh.py`, `python 88.10.PY`** — Python scripts for polynomial regression analysis of surface roughness (Ra) in machining. Use `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`. These are console scripts that produce text output and matplotlib plots; they are not web apps.
- **`kết quả phân tích.docx`** — Word document with analysis results.
- **`README.md`** — Currently contains a copy of `88.05.02.py` (appears mislabeled).

## Running in the sandbox

The web preview is served by `docker-compose.base44.yml` using nginx:alpine on port 3000. The repo root is bind-mounted read-only into the container at `/usr/share/nginx/html`. `index.html` (a copy of `soat.txt`) is served as the default page.

No external credentials or secrets are required.

## Notes

- The Python scripts are not served by the web preview. To run them you would need a Python environment with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `scipy` installed.
- `soat.txt` and `index.html` are kept in sync; edit `soat.txt` and re-copy, or edit `index.html` directly.
