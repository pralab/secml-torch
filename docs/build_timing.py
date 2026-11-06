"""Sphinx extension that logs how long each document takes to read.

MyST-NB executes notebooks while the document is being read, so the time between
``source-read`` and ``doctree-read`` includes the notebook execution.
"""

import time

from sphinx.util import logging

logger = logging.getLogger(__name__)

_started = {}
_elapsed = {}


def _on_source_read(app, docname, source):
    _started[docname] = time.perf_counter()
    logger.info("[timing] start reading %s", docname)


def _on_doctree_read(app, doctree):
    docname = app.env.docname
    start = _started.pop(docname, None)
    if start is None:
        return
    _elapsed[docname] = time.perf_counter() - start
    logger.info("[timing] done  %s in %.1fs", docname, _elapsed[docname])


def _on_build_finished(app, exception):
    if not _elapsed:
        return
    logger.info("[timing] ---- slowest documents ----")
    for docname, seconds in sorted(_elapsed.items(), key=lambda kv: -kv[1]):
        logger.info("[timing] %8.1fs  %s", seconds, docname)
    logger.info("[timing] %8.1fs  total", sum(_elapsed.values()))


def setup(app):
    app.connect("source-read", _on_source_read)
    app.connect("doctree-read", _on_doctree_read)
    app.connect("build-finished", _on_build_finished)
    return {"version": "0.1", "parallel_read_safe": False}
