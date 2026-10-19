"""Paired comparison P asked for: the library run against the paper run with
the SAME run_id, on the six targets, run_ids 0-4, eps = 0.

Both see the same data (DatasetLoader with seed 42 + run_id), so the pairing is
exact and the comparison is per run, not per cell.
"""
import csv, collections
TARGETS = ['ndvi', 'awei_sh', 'evi2', 'hubble', 'n4', 'n11']
RUNS = ['0', '1', '2', '3', '4']
paper = {}
for r in csv.DictReader(open('paper_provenance/verdicts_new_dragonsr.csv')):
    if r['eps'] == '0' and r['target'] in TARGETS and r['run'] in RUNS:
        paper[(r['target'], r['run'])] = r
lib = {}
try:
    for r in csv.DictReader(open('paper_provenance/verdicts_libval.csv')):
        lib[(r['target'], r['run'])] = r
except FileNotFoundError:
    pass
print('%-9s %-4s %-22s %-22s %s' % ('target', 'run', 'paper (joint, eps=0)', 'library', 'agree'))
agree = differ = pending = 0
for t in TARGETS:
    for run in RUNS:
        p = paper.get((t, run)); l = lib.get((t, run))
        pv = (p['hand_verdict'] + ('*' if p.get('star_small_term') else '')) if p else '--'
        if not l:
            lv, note = '(not yet run)', ''
            pending += 1
        elif not l['hand_verdict']:
            lv, note = '(to read)', ''
            pending += 1
        else:
            lv = l['hand_verdict'] + ('*' if l.get('star_small_term') else '')
            same = (p and p['hand_verdict'] == l['hand_verdict'])
            note = 'yes' if same else 'NO'
            agree += bool(same); differ += (not same)
        print('%-9s %-4s %-22s %-22s %s' % (t, run, pv, lv, note if l and l['hand_verdict'] else ''))
print()
print('pairs judged on both sides: %d   agree %d   differ %d   still pending %d'
      % (agree + differ, agree, differ, pending))
pc = collections.Counter()
for t in TARGETS:
    for run in RUNS:
        p = paper.get((t, run))
        if p: pc[p['hand_verdict']] += 1
print('paper side over these 30 pairs: %s' % dict(pc))
