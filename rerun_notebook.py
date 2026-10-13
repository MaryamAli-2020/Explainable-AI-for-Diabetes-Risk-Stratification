"""Execute the complete notebook with fresh imports and save outputs, including on failure."""
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "notebooks/01_diabetes_healthcare_analytics.ipynb"
notebook = nbformat.read(path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

def progress(cell, cell_index, **kwargs):
    print(f"Executing cell {cell_index}: {cell.source.splitlines()[0] if cell.source else '(empty)'}", flush=True)

notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
client = NotebookClient(notebook, kernel_name="python3", timeout=None, resources={"metadata": {"path": str(root)}}, on_cell_start=progress)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print("Notebook completed successfully.", flush=True)
