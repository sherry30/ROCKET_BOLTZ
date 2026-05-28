# cap_bfactors.py
import sys

input_pdb = sys.argv[1]
output_pdb = sys.argv[2]
b_cap = float(sys.argv[3]) if len(sys.argv) > 3 else 80.0

with open(input_pdb) as fin, open(output_pdb, 'w') as fout:
    for line in fin:
        if line.startswith(('ATOM', 'HETATM')):
            # B-factor is columns 61-66 (1-indexed), so [60:66] in Python
            b_str = line[60:66]
            try:
                b = float(b_str)
                if b > b_cap:
                    b = b_cap
                line = line[:60] + f'{b:6.2f}' + line[66:]
            except ValueError:
                pass
        fout.write(line)

print(f'Wrote {output_pdb} with B-factors capped at {b_cap}')