"""
data.py — Download and parse RNA structures from the Protein Data Bank (PDB)

The PDB is the world repository of all experimentally solved molecular structures.
This module downloads .cif files and extracts C1' atom coordinates for each nucleotide.

Key concept: C1' is the sugar carbon connecting the base to the RNA backbone.
We track only this atom (one per nucleotide) to represent the 3D structure.
"""

import requests
import time
import pandas as pd
from Bio.PDB import MMCIFParser

# Maps modified/non-standard nucleotides back to canonical A/U/G/C
CANONICAL_MAP = {
    '2MG':'G', 'H2U':'U', 'PSU':'U', '5MC':'C', '1MA':'A',
    'M2G':'G', '7MG':'G', '5MU':'U', 'YYG':'G', 'GTP':'G',
    'ATP':'A', 'CTP':'C', 'UTP':'U', 'GDP':'G', 'ADP':'A',
    'OMG':'G', 'OMC':'C', 'OMA':'A', 'OMU':'U', '5BU':'U',
}

def download_structure(pdb_id, save_dir="."):
    """Download a single .cif file from PDB. Returns True if successful."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return False
        with open(f"{save_dir}/{pdb_id}.cif", "w") as f:
            f.write(r.text)
        return True
    except Exception:
        return False


def parse_structure(pdb_id, cif_path):
    """
    Parse a .cif file and extract C1' coordinates.

    Returns a DataFrame with columns:
        chain, resid, resname (A/U/G/C), x, y, z
    Returns None if parsing fails or no RNA found.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception:
        return None

    rows = []
    for model in structure:
        for chain in model:
            for res in chain:
                rn = CANONICAL_MAP.get(
                    res.get_resname().strip(),
                    res.get_resname().strip())
                if rn not in ['A','U','G','C']:
                    continue
                if "C1'" not in res:
                    continue
                x, y, z = res["C1'"].get_vector()
                rows.append({
                    'chain'  : chain.id,
                    'resid'  : res.get_id()[1],
                    'resname': rn,
                    'x': float(x), 'y': float(y), 'z': float(z)
                })
        break  # first model only

    if not rows:
        return None
    return pd.DataFrame(rows)


def bulk_download(pdb_ids, max_structures=100, save_dir="."):
    """
    Download and parse multiple PDB structures.
    Includes rate limiting to avoid overwhelming PDB servers.

    Returns dict: {pdb_id: DataFrame}
    """
    structures = {}
    failed     = 0

    for i, pdb_id in enumerate(pdb_ids[:max_structures]):
        if i % 20 == 0 and i > 0:
            print(f"  {i}/{max_structures} loaded: {len(structures)}")
            time.sleep(1)

        if not download_structure(pdb_id, save_dir):
            failed += 1
            continue

        df = parse_structure(pdb_id, f"{save_dir}/{pdb_id}.cif")
        if df is not None and len(df) >= 15:
            structures[pdb_id] = df
        else:
            failed += 1

    print(f"Loaded: {len(structures)}, Failed: {failed}")
    return structures
