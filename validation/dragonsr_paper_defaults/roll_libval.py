"""Collect and screen the library-validation runs.

These are the packaged DragonSR estimator (branch `dragonsr`, commit
681afa22) run at the paper's defaults on six targets x five run_ids at
eps = 0, by the paper session. They are judged HERE, by the experiment
session, so that the session which wrote the library is not the one
certifying it.

Same rules as every other arm: residual-refuted rows are NO without reading;
passers are read by hand. The residual sample is the usual one -- a fresh
draw for the synthetic targets, the same pixels for the indices -- not the
run's own training data.

Writes paper_provenance/verdicts_libval.csv, arm `library_validation`.
Touches no other verdict file.
"""
import csv, json, glob, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.abspath('..'))
import lib.dragon; sys.modules['dragon'] = sys.modules['lib.dragon']
from dataprocessing.Data import PhysicsGenerator, NguyenGenerator, RemoteSensingLoader
RES = '/scratch/jkeisler/dragonsr_lib/results'
OUT = 'paper_provenance/verdicts_libval.csv'
F = ['engine', 'arm', 'target', 'eps', 'run', 'batch', 'path', 'judged_formula',
     'selection_rule', 'hand_verdict', 'hand_reason', 'star_small_term',
     'guard_active', 'evaluated_is_law', 'resid_rel', 'resid_rms', 'tol',
     'numeric', 'actualT', 'winner_type', 'stream', 'library_commit']
PHYS = {'hubble'}
NGY = {'n4', 'n11'}
ENV = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "sqrt": np.sqrt,
       "abs": np.abs, "log": np.log, "ln": np.log, "log10": np.log10,
       "sign": np.sign, "where": np.where, "minimum": np.minimum,
       "maximum": np.maximum, "clip": np.clip}
_c = {}
def sample(t):
    if t not in _c:
        _c[t] = (PhysicsGenerator(4000, 99991).generate(t) if t in PHYS else
                 NguyenGenerator(4000, 99991).generate(t) if t in NGY else
                 RemoteSensingLoader().load(t))
    return _c[t]
have = {}
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        have[(r['target'], r['run'])] = r
added = 0
for p in sorted(glob.glob(os.path.join(RES, '*_run*.json'))):
    m = re.match(r'(.+)_run(\d+)\.json$', os.path.basename(p))
    if not m:
        continue
    t, run = m.group(1), m.group(2)
    if (t, run) in have:
        continue
    try:
        j = json.load(open(p))
    except Exception:
        continue                      # still being written
    d = {c: '' for c in F}
    d.update(engine='dragonsr', arm='library_validation', target=t, eps='0',
             run=run, batch='dragonsr_lib', path=os.path.abspath(p),
             selection_rule='DragonSR.formula_ as the library reports it',
             actualT=str(j.get('n_evaluations') or ''),
             winner_type=str(j.get('winner_type') or ''),
             stream=str(j.get('stream') or ''),
             library_commit=str(j.get('library_commit') or ''))
    f = str(j.get('formula') or '')
    d['judged_formula'] = f
    try:
        X, y = sample(t)
        yv = np.asarray(y, float)
        sd = float(yv.std()); rms = float(np.sqrt(np.mean(yv**2)))
        e = dict(ENV)
        for c in X.columns:
            e[c] = X[c].to_numpy(float)
        with np.errstate(all='ignore'):
            pr = np.asarray(eval(f.replace('^', '**'), {'__builtins__': {}}, e), float)
        if np.ndim(pr) == 0:
            pr = np.full_like(yv, float(pr))
        ok = np.isfinite(pr) & np.isfinite(yv)
        if ok.sum() < 0.5 * len(yv):
            d['numeric'] = 'nonfinite'
        else:
            err = float(np.sqrt(np.mean((pr[ok] - yv[ok]) ** 2)))
            tol = 0.01
            d['resid_rel'] = '%.6e' % (err / sd)
            d['resid_rms'] = '%.6e' % (err / rms)
            d['tol'] = '%.6e' % tol
            d['numeric'] = 'pass' if err / rms <= tol else 'fail'
    except Exception as ex:
        d['numeric'] = 'err:' + type(ex).__name__
    if d['numeric'] in ('fail', 'nonfinite'):
        d['hand_verdict'] = 'NO'
        d['hand_reason'] = ('residual outside the stated band (numeric=%s); decided '
                            'without reading' % d['numeric'])
    have[(t, run)] = d; added += 1
rows = sorted(have.values(), key=lambda r: (r['target'], int(r['run'])))
for r in rows:
    for c in F:
        r.setdefault(c, '')
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=F); w.writeheader(); w.writerows(rows)
c = collections.Counter(r['target'] for r in rows)
print('library_validation: %d rows (added %d) over %d targets' % (len(rows), added, len(c)))
print('   per target: %s' % dict(c))
ev = [int(r['actualT']) for r in rows if r['actualT']]
if ev:
    print('   n_evaluations: %d-%d' % (min(ev), max(ev)))
print('   TO READ: %d' % sum(1 for r in rows if r['numeric'] == 'pass' and not r['hand_verdict']))
for r in rows:
    if r['numeric'] == 'pass' and not r['hand_verdict']:
        print('   >> %-8s r%s resid_rms %s  [%s]' % (r['target'], r['run'],
                                                     r['resid_rms'], r['winner_type']))
        print('      %s' % r['judged_formula'][:160])
