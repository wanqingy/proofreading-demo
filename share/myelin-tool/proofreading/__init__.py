"""Myelin-tagging tool package (trimmed copy -- see share/myelin-tool/sync.py).

The full repo's ``proofreading/__init__.py`` imports the notebook fly-through toolkit
(``.viewer`` / ``.skeleton`` / ``.flythrough`` / ``.context``), which needs ``neuroglancer`` +
``ipywidgets`` + (optionally) ``navis``. This tool never uses that toolkit -- it drives
neuroglancer entirely from the browser (``web/src/myelin.ts``) -- so this stub replaces it
rather than being a copy, keeping this package's Python dependency list to exactly what
``proofreading.em.serve`` needs: numpy, caveclient, cloud-volume, fastapi, uvicorn.
"""
