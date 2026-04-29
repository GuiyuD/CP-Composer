#!/usr/bin/env python3
"""Batch run CP-Composer codesign (condition2/head-to-tail setting) from protein-ligand PDB files.

For each input PDB:
1) detect pocket residues on receptor chain by CA-within-threshold to any ligand atom.
2) write pocket json compatible with api.run.
3) run `python -m api.run --mode codesign` with peptide length = ligand_length + 2.
   (implemented as [L+2, L+3) because api.run uses np.random.randint(lmin, lmax)).
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_gpu_arg(gpu: int) -> int:
    """Fallback to CPU when CUDA is unavailable in current runtime."""
    if gpu < 0:
        return -1
    try:
        import torch

        if not torch.cuda.is_available():
            print("[WARN] CUDA is not available in this environment, falling back to CPU (--gpu -1).")
            return -1
        if torch.cuda.device_count() <= gpu:
            print(
                f"[WARN] Requested gpu index {gpu}, but only {torch.cuda.device_count()} visible device(s). "
                "Falling back to CPU (--gpu -1)."
            )
            return -1
        return gpu
    except Exception as e:
        print(f"[WARN] Failed to probe CUDA availability ({e}), falling back to CPU (--gpu -1).")
        return -1


def resolve_ckpt(ckpt: str, project_root: Path) -> str:
    if ckpt is None:
        return None
    ckpt_path = Path(ckpt).expanduser().resolve()
    if ckpt_path.exists():
        return str(ckpt_path)

    candidates = sorted((project_root / "checkpoints").glob("**/*.ckpt"))
    hint = "\n".join(f"- {c}" for c in candidates[:20]) if candidates else "(no .ckpt found under checkpoints/)"
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}\n"
        f"Please provide a valid --ckpt path.\n"
        f"Detected checkpoint candidates under {project_root / 'checkpoints'}:\n{hint}"
    )


def parse_pdb_atoms(pdb_path: Path):
    atoms = []
    with pdb_path.open("r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec != "ATOM":
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atoms.append(
                {
                    "atom_name": atom_name,
                    "resname": resname,
                    "chain": chain,
                    "resseq": resseq,
                    "icode": icode,
                    "coord": (x, y, z),
                }
            )
    return atoms


def sqdist(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def build_residue_order(atoms, chain_id: str) -> List[Tuple[str, str]]:
    seen = []
    seen_set = set()
    for a in atoms:
        if a["chain"] != chain_id:
            continue
        key = (a["resseq"], a["icode"])
        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)
    return seen


def ligand_length(atoms, ligand_chain: str) -> int:
    return len(build_residue_order(atoms, ligand_chain))


def pocket_by_ca_to_ligand(
    atoms, receptor_chain: str, ligand_chain: str, cutoff: float
) -> List[Tuple[str, Tuple[int, str]]]:
    cutoff2 = cutoff * cutoff
    lig_coords = [a["coord"] for a in atoms if a["chain"] == ligand_chain]
    if not lig_coords:
        raise ValueError(f"No atoms found in ligand chain {ligand_chain}")

    # collect receptor CA by residue order
    rec_residues = build_residue_order(atoms, receptor_chain)
    ca_by_res: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    for a in atoms:
        if a["chain"] == receptor_chain and a["atom_name"] == "CA":
            key = (a["resseq"], a["icode"])
            ca_by_res[key] = a["coord"]

    pocket = []
    for res in rec_residues:
        if res not in ca_by_res:
            continue
        ca = ca_by_res[res]
        if any(sqdist(ca, lc) <= cutoff2 for lc in lig_coords):
            # api.run expects (chain, (resseq, icode)) style as produced by detect_pocket
            rid = (int(res[0]), res[1] if res[1] else " ")
            pocket.append((receptor_chain, rid))

    if not pocket:
        raise ValueError(
            f"No receptor residues within {cutoff}A (CA) to ligand chain {ligand_chain}"
        )
    return pocket


def run_one(
    pdb_file: Path,
    out_root: Path,
    receptor_chain: str,
    ligand_chain: str,
    cutoff: float,
    gpu: int,
    n_samples: int,
    ckpt: str,
    api_run_py: Path,
    cwd: Path,
):
    pdb_file = pdb_file.resolve()
    out_root = out_root.resolve()
    atoms = parse_pdb_atoms(pdb_file)
    lig_len = ligand_length(atoms, ligand_chain)
    if lig_len <= 0:
        raise ValueError(f"Ligand chain {ligand_chain} has zero residues")

    # fixed generated length = ligand length + 2, via [L+2, L+3)
    length_min = lig_len + 2
    length_max = lig_len + 3

    pocket = pocket_by_ca_to_ligand(atoms, receptor_chain, ligand_chain, cutoff)

    case_dir = (out_root / pdb_file.stem).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    pocket_json = case_dir / "pocket_ca10A.json"
    with pocket_json.open("w") as f:
        json.dump(pocket, f)

    cmd = [
        sys.executable,
        str(api_run_py),
        "--mode",
        "codesign",
        "--pdb",
        str(pdb_file),
        "--pocket",
        str(pocket_json.resolve()),
        "--out_dir",
        str((case_dir / "codesign").resolve()),
        "--length_min",
        str(length_min),
        "--length_max",
        str(length_max),
        "--n_samples",
        str(n_samples),
        "--gpu",
        str(gpu),
    ]
    if ckpt:
        cmd.extend(["--ckpt", ckpt])

    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def resolve_api_run(project_root: Path, strict_project_root: bool = False) -> Path:
    """Resolve api/run.py robustly.

    Priority:
    1) <project_root>/api/run.py
    2) <this_script_parent_parent>/api/run.py
    3) <cwd>/api/run.py
    """
    if strict_project_root:
        candidates = [project_root / "api" / "run.py"]
    else:
        candidates = [
            project_root / "api" / "run.py",
            Path(__file__).resolve().parents[1] / "api" / "run.py",
            Path.cwd() / "api" / "run.py",
        ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "Cannot find api/run.py. Checked:\n"
        + "\n".join(f"- {str(c)}" for c in candidates)
        + "\nPlease set --project_root to CP-Composer repository root."
    )



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True, help="Directory containing input *.pdb files")
    p.add_argument("--out_dir", required=True, help="Output root directory")
    p.add_argument("--receptor_chain", default="R")
    p.add_argument("--ligand_chain", default="L")
    p.add_argument("--cutoff", type=float, default=10.0, help="CA-to-ligand distance cutoff (A)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=100, help="Number of generated peptides per PDB")
    p.add_argument("--project_root", default=None, help="Path to CP-Composer root (defaults to parent of this script)")
    p.add_argument("--ckpt", default=None, help="Path to codesign checkpoint (.ckpt)")
    args = p.parse_args()

    pdb_dir = Path(args.pdb_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    ckpt = resolve_ckpt(args.ckpt, project_root)
    gpu = normalize_gpu_arg(args.gpu)
    api_run_py = resolve_api_run(project_root, strict_project_root=(args.project_root is not None))
    run_cwd = api_run_py.parent.parent
    out_root.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files under {pdb_dir}")

    for pdb_file in pdb_files:
        try:
            run_one(
                pdb_file=pdb_file,
                out_root=out_root,
                receptor_chain=args.receptor_chain,
                ligand_chain=args.ligand_chain,
                cutoff=args.cutoff,
                gpu=gpu,
                n_samples=args.n_samples,
                ckpt=ckpt,
                api_run_py=api_run_py,
                cwd=run_cwd,
            )
        except Exception as e:
            print(f"[ERROR] {pdb_file.name}: {e}")


if __name__ == "__main__":
    main()
