---
title: Switchboard
emoji: 🔀
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
- {{model_repo}}
datasets:
- {{dataset_repo}}
short_description: A learned router between a small and a large LLM
---

# Switchboard demo

Type a prompt and see whether the router sends it to the small or the large model. The
classifier ([{{model_repo}}](https://huggingface.co/{{model_repo}})) runs in the visitor's
browser with transformers.js, so this static Space has no server, makes no LLM calls and holds
no API key.

> {{headline}}

Code: {{github_url}}
