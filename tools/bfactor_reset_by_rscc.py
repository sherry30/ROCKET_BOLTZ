# bfactor_reset_by_rscc.py
import sys
import re

input_pdb = sys.argv[1]
cc_log = sys.argv[2]  # path to cc_per_residue.log
output_pdb = sys.argv[3]

# Thresholds — tune these
CC_HIGH = 0.7  # residues with CC above this → low B
CC_LOW = 0.3  # residues with CC below this → high B / delete

B_HIGH_QUALITY = 30.0  # for residues fitting density well
B_MEDIUM = 60.0  # for in-between residues
B_POOR = 100.0  # for residues fitting density poorly
DELETE_BELOW_CC = 0.0  # set to None to disable deletion; set to e.g. 0.1 to delete near-zero-CC

# Parse the cc_per_residue.log — format depends on phenix version
# Typical format: one line per residue, columns include chain, resseq, cc
# We need to find the lines and extract (chain, resseq) -> cc

residue_cc = {}
with open(cc_log) as f:
    for line in f:
        # The format varies; try a few patterns
        # Pattern: 'A   VAL    1   1.00  225.24  0.2112   0.92   0.33'
        m = re.match(r'\s*(\w)\s+\w{3}\s+(-?\d+)\s+[\d.]+\s+[\d.]+\s+(-?[\d.]+)', line)
        if m:
            chain, resseq, cc = m.group(1), int(m.group(2)), float(m.group(3))
            residue_cc[(chain, resseq)] = cc

print(f'Parsed CC for {len(residue_cc)} residues from {cc_log}')

# Sanity check
if len(residue_cc) == 0:
    print('WARNING: no residues parsed — check the log format')
    print('Showing first 5 lines of log:')
    with open(cc_log) as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f'  {line.rstrip()}')
    sys.exit(1)

# Now rewrite the PDB with new B-factors
n_high, n_med, n_poor, n_del, n_missing = 0, 0, 0, 0, 0

with open(input_pdb) as fin, open(output_pdb, 'w') as fout:
    for line in fin:
        if line.startswith(('ATOM', 'HETATM')):
            # Standard PDB columns:
            # 22 (zero-indexed 21): chain ID
            # 23-26 (zero-indexed 22:26): residue sequence number
            # 61-66 (zero-indexed 60:66): B-factor
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                fout.write(line)
                continue

            cc = residue_cc.get((chain, resseq))
            if cc is None:
                # Residue not in log — keep original B
                n_missing += 1
                fout.write(line)
                continue

            if DELETE_BELOW_CC is not None and cc < DELETE_BELOW_CC:
                # Skip this atom entirely
                n_del += 1
                continue

            # Pick new B-factor by CC
            if cc > CC_HIGH:
                b_new = B_HIGH_QUALITY
                n_high += 1
            elif cc > CC_LOW:
                b_new = B_MEDIUM
                n_med += 1
            else:
                b_new = B_POOR
                n_poor += 1

            line = line[:60] + f'{b_new:6.2f}' + line[66:]

        fout.write(line)

n_total = n_high + n_med + n_poor + n_del
print(f'Processed {n_total} atoms (+ {n_missing} unchanged, no CC):')
print(f'  CC > {CC_HIGH}     → B = {B_HIGH_QUALITY}: {n_high} atoms')
print(f'  {CC_LOW} < CC ≤ {CC_HIGH}  → B = {B_MEDIUM}: {n_med} atoms')
print(f'  CC ≤ {CC_LOW}      → B = {B_POOR}: {n_poor} atoms')
if DELETE_BELOW_CC is not None:
    print(f'  CC < {DELETE_BELOW_CC}      → deleted: {n_del} atoms')
print(f'Wrote {output_pdb}')