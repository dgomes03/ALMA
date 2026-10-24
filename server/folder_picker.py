"""Native folder picker for the web UI's "select folder to index" button.

macOS subtlety: Anaconda/conda Python binaries are CLI-only and cannot show GUI
windows. We locate a Python.app framework binary (a proper macOS application bundle)
and run the tkinter picker in that subprocess so the dialog actually appears.
Falls back to the running interpreter on Windows/Linux.
"""
import glob
import os
import platform
import subprocess
import sys
import tempfile


_PICKER_SCRIPT = "\n".join([
    "import tkinter as tk",
    "from tkinter import filedialog",
    "root = tk.Tk()",
    "root.withdraw()",
    "root.lift()",
    "root.attributes('-topmost', True)",
    "root.focus_force()",
    "path = filedialog.askdirectory(title='Select folder to index')",
    "root.destroy()",
])


def _find_gui_python() -> str:
    """On macOS, locate a Python.app framework binary that can host GUI windows.
    Returns sys.executable on other platforms or when no candidate works."""
    if platform.system() != 'Darwin':
        return sys.executable

    candidates = []
    path = sys.executable
    for _ in range(6):
        path = os.path.dirname(path)
        p = os.path.join(path, 'python.app', 'Contents', 'MacOS', 'python')
        if os.path.exists(p):
            candidates.append(p)

    for fixed in [
        '/opt/anaconda3/python.app/Contents/MacOS/python',
        os.path.expanduser('~/anaconda3/python.app/Contents/MacOS/python'),
        os.path.expanduser('~/miniconda3/python.app/Contents/MacOS/python'),
    ]:
        if os.path.exists(fixed) and fixed not in candidates:
            candidates.append(fixed)

    candidates += sorted(glob.glob(
        '/opt/homebrew/opt/python*/Frameworks/Python.framework/'
        'Versions/Current/Resources/Python.app/Contents/MacOS/Python'
    ), reverse=True)

    for c in candidates:
        try:
            check = subprocess.run([c, '-c', 'import tkinter'], capture_output=True, timeout=5)
            if check.returncode == 0:
                return c
        except Exception:
            continue
    return sys.executable


def pick_folder() -> dict:
    """Show a native folder picker. Returns {'path': str, 'cancelled': bool, 'error'?: str}."""
    python_exe = _find_gui_python()
    result_path = tempfile.mktemp(suffix='_rag_picker.txt')

    # Write result to a temp file rather than capturing stdout — capture_output=True pipes
    # stdout/stderr, which macOS treats as "non-interactive process" and blocks GUI windows.
    script = _PICKER_SCRIPT + (
        f"\ntry:\n"
        f"    open({result_path!r}, 'w', encoding='utf-8').write(path or '')\n"
        f"except Exception:\n"
        f"    pass\n"
    )

    try:
        proc = subprocess.Popen([python_exe, '-c', script])
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {'path': '', 'cancelled': True}
    except Exception as e:
        return {'path': '', 'cancelled': False, 'error': str(e)}

    selected = ''
    if os.path.exists(result_path):
        with open(result_path, encoding='utf-8') as f:
            selected = f.read().strip()
        os.unlink(result_path)

    return {'path': selected, 'cancelled': selected == ''}
