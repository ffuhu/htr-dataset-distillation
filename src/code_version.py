"""
Code traceability. The commit is fixed locally and travels in the file CODE_VERSION (repository root,
not versioned), so that runs on machines where the code is copied rather than checked out can be traced:
	line 1: "commit <hash>"; rest: "<md5>  <path>" for every versioned file in src/ configs/ slurm/ scripts_gen/.

- `python src/code_version.py write` (clean tree): writes CODE_VERSION.
- `stamp()` (in every run): checks the md5 of all listed files and returns the commit; if CODE_VERSION is missing
  or any file does not match, it raises an error (the run does not correspond to any commit).
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "CODE_VERSION"
TRACKED_DIRS = ["src", "configs", "slurm", "scripts_gen"]


def _digest(path: Path) -> str:
	if path.is_symlink():  # e.g. src/models/utils: the link target is versioned, not its contents
		return hashlib.md5(os.readlink(path).encode()).hexdigest()
	return hashlib.md5(path.read_bytes()).hexdigest()


def write():
	dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
						capture_output=True, text=True, check=True).stdout
	if dirty:
		sys.exit(f"Árbol con cambios sin commitear; haz commit antes de desplegar:\n{dirty}")
	commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
	files = subprocess.run(["git", "ls-files", *TRACKED_DIRS], cwd=ROOT, capture_output=True, text=True,
						check=True).stdout.splitlines()
	lines = [f"commit {commit}"] + [f"{_digest(ROOT / f)}  {f}" for f in files]
	VERSION_FILE.write_text("\n".join(lines) + "\n")
	print(f"CODE_VERSION: commit {commit}, {len(files)} ficheros")


def stamp() -> str:
	if not VERSION_FILE.exists():
		raise RuntimeError(f"Falta {VERSION_FILE}: despliega el código con scripts_gen/deploy_code.sh")
	head, *entries = VERSION_FILE.read_text().splitlines()
	commit = head.split()[1]
	bad = []
	for e in entries:
		md5, rel = e.split("  ", 1)
		p = ROOT / rel
		if not (p.exists() or p.is_symlink()) or _digest(p) != md5:
			bad.append(rel)
	if bad:
		raise RuntimeError(f"El código difiere del commit {commit} en {len(bad)} ficheros: {bad[:10]}")
	print(f"Código: commit {commit} ({len(entries)} ficheros verificados)")
	return commit


if __name__ == "__main__":
	if sys.argv[1:] == ["write"]:
		write()
	elif sys.argv[1:] == ["check"]:
		stamp()
	else:
		sys.exit("uso: python src/code_version.py write|check")
