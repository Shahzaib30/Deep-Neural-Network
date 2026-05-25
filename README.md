# Deep Neural Network — Projects and Notebooks

This folder contains small experiments and example projects exploring neural
network techniques (PyTorch). To keep the repo easy to navigate, move Jupyter
notebooks into `notebooks/` and Python projects into `projects/`.

Quick actions

- Reorganize files (move notebooks -> `notebooks/`, `.py` -> `projects/`):

```bash
python tools/reorganize_deep_nn.py
```

After running the script, inspect `projects/` and `notebooks/` for the new
structure. Each Python project moved gets a small `README.md` derived from
the module docstring if present.

Notes

- Notebooks are preserved and moved; no content is changed.
- The script will skip the `data/` and `.vscode/` folders.
- Review and commit the reorganized layout after verification.

If you'd like, I can run the reorganizer now and then polish generated
project READMEs; tell me to proceed and I'll run it and report results.
