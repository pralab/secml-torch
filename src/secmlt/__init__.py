# noqa: D104

import pathlib

version_path = pathlib.Path(__file__).parent / "VERSION"

with version_path.open() as f:
    __version__ = f.read()
