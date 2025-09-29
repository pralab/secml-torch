import os
import sys

# Add the library path: assumes 'docs/' is in the root, and 'src/' is sibling
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
)

project = "SecML-Torch"
