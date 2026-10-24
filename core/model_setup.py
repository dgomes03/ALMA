"""Finding the language model at startup, and downloading one if there is none.

MLX models live in the `model` folder of the project (git-ignored). With none
there, the user picks one of the Ministral 3 models (4-bit) and it downloads
from Hugging Face into that folder. On the Ollama backend the same choice pulls
the model into Ollama and saves it as OLLAMA_MODEL.
"""
import json
import os
import shutil
import sys
from dataclasses import dataclass

from core.config import MODEL_PATH, PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


@dataclass(frozen=True)
class ModelChoice:
    name: str
    mlx_repo: str
    mlx_gb: float
    ollama_tag: str
    ollama_gb: float
    best_for: str


CHOICES = (
    ModelChoice("Ministral 3 3B", "mlx-community/Ministral-3-3B-Instruct-2512-4bit", 2.8,
                "hf.co/lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF:Q4_K_M", 2.1,
                "For very simple, quick questions. Fast, but more likely to hallucinate. Works with 8 GB of memory."),
    ModelChoice("Ministral 3 8B", "mlx-community/Ministral-3-8B-Instruct-2512-4bit", 5.6,
                "hf.co/lmstudio-community/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M", 5.2,
                "For questions of medium complexity. Handles more complex searches. Needs 16 GB of memory."),
    ModelChoice("Ministral 3 14B", "mlx-community/Ministral-3-14B-Instruct-2512-4bit", 8.5,
                "hf.co/lmstudio-community/Ministral-3-14B-Instruct-2512-GGUF:Q4_K_M", 8.2,
                "The minimum recommended for an accurate RAG system with few hallucinations. Searches long "
                "documents and reasons over them. Needs 16 to 24 GB of memory."),
)
RECOMMENDED = 2          # the 14B, where the cursor starts


def installed_models():
    """Model folders in `model/` (each holds an MLX model's config.json)."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return [os.path.join(MODEL_DIR, name) for name in sorted(os.listdir(MODEL_DIR))
            if not name.endswith(".partial")
            and os.path.isfile(os.path.join(MODEL_DIR, name, "config.json"))]


def _choose_download(backend):
    """Ask which Ministral 3 model to get. Returns a ModelChoice."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise RuntimeError(
            "No language model found. Start ALMA in a terminal to choose one, or put an "
            f"MLX model folder in {MODEL_DIR}.")
    from core.config_menu import pick
    size = (lambda c: c.mlx_gb) if backend == "mlx" else (lambda c: c.ollama_gb)
    rows = [(c.name, f"{size(c):.1f} GB download") for c in CHOICES]
    header = ("No language model is installed yet. Choose one to download (4-bit).",
              "You can switch later by adding other models to the model folder or in the settings menu.")
    index = pick("Choose a language model", rows, details=[c.best_for for c in CHOICES],
                 header=header, index=RECOMMENDED)
    if index is None:
        raise SystemExit("No model chosen. ALMA needs a language model to run.")
    return CHOICES[index]


def mlx_model_path():
    """The MLX model to load: MODEL_PATH if set, else the model in `model/`,
    asking which one when there are several and downloading one when there are none."""
    if MODEL_PATH:
        local = MODEL_PATH if os.path.isabs(MODEL_PATH) else os.path.join(PROJECT_ROOT, MODEL_PATH)
        return local if os.path.isdir(local) else MODEL_PATH       # else a Hugging Face repo id
    found = installed_models()
    if len(found) == 1:
        return found[0]
    if found:
        from core.config_menu import pick
        index = pick("Choose the model to use", [(os.path.basename(p), "") for p in found],
                     header=("There are several models in the model folder.",))
        if index is None:
            raise SystemExit("No model chosen.")
        return found[index]
    return download_mlx(_choose_download("mlx"))


def download_mlx(choice):
    """Download `choice` into model/<name>. It lands in a .partial folder first,
    so an interrupted download is never mistaken for a working model."""
    from huggingface_hub import snapshot_download
    target = os.path.join(MODEL_DIR, choice.mlx_repo.split("/")[-1])
    partial = target + ".partial"
    print(f"\nDownloading {choice.name} ({choice.mlx_gb:.1f} GB) into the model folder. "
          f"This happens once; an interrupted download resumes next time.\n")
    snapshot_download(choice.mlx_repo, local_dir=partial)
    shutil.move(partial, target)
    return target


def ensure_ollama_model(model):
    """Make sure Ollama has `model.name`; if not, ask which model to pull,
    pull it with a progress line, and save the choice as OLLAMA_MODEL."""
    if model.has_model():
        return model.name
    choice = _choose_download("ollama")
    print(f"\nPulling {choice.name} ({choice.ollama_gb:.1f} GB) into Ollama. This happens once.\n")
    with model.session.post(f"{model.host}/api/pull", json={"model": choice.ollama_tag, "stream": True},
                            stream=True, timeout=(10, None)) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            status = json.loads(line)
            if status.get("error"):
                raise RuntimeError(f"Ollama could not pull {choice.ollama_tag}: {status['error']}")
            done, total = status.get("completed"), status.get("total")
            progress = f" {100 * done / total:5.1f}%" if done and total else ""
            print(f"\r  {status.get('status', '')[:50]:<50}{progress}", end="", flush=True)
    print()
    from core.config_menu import write_changes
    write_changes({"OLLAMA_MODEL": repr(choice.ollama_tag)})
    return choice.ollama_tag
