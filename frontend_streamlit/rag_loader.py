import importlib.util
from pathlib import Path
import sys


def load_module_from_path(name: str, path: Path):
    path = Path(path)
    parent_dir = str(path.parent)

    inserted = False
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        inserted = True

    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and parent_dir in sys.path:
            try:
                sys.path.remove(parent_dir)
            except ValueError:
                pass


def load_rag_helpers():
    root = Path(__file__).resolve().parents[1]
    rag_dir = root / "Rag"

    main_path = rag_dir / "main.py"
    generator_path = rag_dir / "generator.py"

    rag_main = load_module_from_path("rag_main", main_path)
    rag_generator = load_module_from_path("rag_generator", generator_path)

    # Expose what we need from the loaded modules
    build_pipeline = getattr(rag_main, "build_pipeline")
    answer_query = getattr(rag_main, "answer_query")
    GenericLLM = getattr(rag_generator, "GenericLLM")

    return build_pipeline, answer_query, GenericLLM
