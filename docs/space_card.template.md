---
title: Switchboard
emoji: 🔀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
models:
- {{model_repo}}
datasets:
- {{dataset_repo}}
short_description: A learned router between a small and a large LLM
---

# Switchboard demo

Type a prompt and see whether the router sends it to the small or the large model. The Space
only runs the classifier ([{{model_repo}}](https://huggingface.co/{{model_repo}})); it makes no
LLM calls and holds no API key.

> {{headline}}

Code: {{github_url}}
