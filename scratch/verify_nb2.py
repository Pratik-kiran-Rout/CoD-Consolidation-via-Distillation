import json

nb = json.load(open('Notebook_Codes/phase_2/phase_2.ipynb', encoding='utf-8'))
print('Total cells in phase_2.ipynb:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    if ctype == 'markdown':
        s = c['source'][0].strip() if c['source'] else 'empty'
        print(f"Cell {i} [markdown]: {s[:60]}")
    elif ctype == 'code':
        outs = len(c.get('outputs', []))
        print(f"Cell {i} [code]: outputs={outs}, lines={len(c['source'])}")

