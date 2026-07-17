# html_builder.py
from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

from Config import Dragon as _CfgDragon, PySR as _CfgPySR, Experiment as _CfgExp
from helpers.stats import _make_formula_stats_svg


# ══════════════════════════════════════════════════════════════════════════════
#  HTML LEADERBOARD BUILDER
# ══════════════════════════════════════════════════════════════════════════════


_WINNER_REMAP = {
    "rational": "polyrat",
    "nested":   "nested",
    "ols":      "ols",
    "channel":  "channel",
}

_INIT_STRATEGY_LABELS = {
  "random": "Random uniform",
  "diverse": "Diverse population (seed DAGs)",
  "xgboost": "XGBoost feature select",
  "warmstart": "Warm-start (PySR)",
  "adversarial": "Adversarial init",
}


def _result_to_db_entry(r):
    """Convert a dragon_worker / run_pysr result dict to HTML DB entry format."""
    loss = r.get("loss")
    finite_loss = (float(loss) if loss is not None and np.isfinite(float(loss)) else None)
    wt   = _WINNER_REMAP.get(r.get("winner_type", ""), None)
    rd   = r.get("rat_degree")
    is_pysr = r.get("method") in ("pysr", "pysr_noise")
    search_loss_name = str(
      r.get("search_loss") or ("mse" if is_pysr else "corr")
    ).strip().lower()
    if search_loss_name == "channel":
      search_loss_name = "corr"
    search_loss_kind = {
      "corr": "1−|corr|",
      "mse": "MSE",
      "raw_mse": "Raw MSE",
      "mae": "MAE",
      "huber": "Huber",
    }.get(search_loss_name, search_loss_name.upper())
    if is_pysr:
        total_t = _CfgPySR.N_ITERATIONS
        pop_k   = _CfgPySR.POPULATION_SIZE
        # Max complexity actually reached on the Pareto frontier.
        _hof    = r.get("hall_of_fame") or []
        _cplx   = [h.get("complexity") for h in _hof if h.get("complexity") is not None]
        max_c   = max(_cplx) if _cplx else 20  # 20 = PySR options.maxsize cap
    else:
        total_t = int(r.get("actualT") if r.get("actualT") is not None else _CfgDragon.N_ITERATIONS)
        pop_k   = _CfgDragon.K_INIT
        _mc = r.get("max_complexity_reached")
        max_c = int(_mc) if _mc is not None and pd.notna(_mc) else _CfgDragon.MAX_COMPLEXITY
    # Smart-parallel extras (only present when method == 'spar').
    actual_t_winner     = r.get("actualT_winner")
    actual_t_per_stream = r.get("actualT_per_stream") or {}
    return {
        "oneMinusR2": round(finite_loss, 8) if finite_loss is not None else None,
        "mse":        None,
        "winner":     wt,
        "bestCh":     None,
        "polyDeg":    int(rd) if rd is not None and pd.notna(rd) else None,
        "nestedLink": None,
        "finalExpr":  str(r.get("formula", "")),
        "runtime":    round(float(r.get("time_s", 0)), 2),
        "totalT":     total_t,
        "actualTWinner":   int(actual_t_winner) if actual_t_winner is not None else None,
        "actualTPerStream": {str(k): int(v) for k, v in actual_t_per_stream.items()},
        "K":          pop_k,
        "maxComp":    max_c,
        "phase":      None,
        # Min search loss reached (the actual quantity the search optimised).
        # Dragon -> alignment_loss (1 - |corr|);  PySR -> raw MSE returned per member.
        "searchLoss": finite_loss,
        "searchLossKind": search_loss_kind,
        "searchLossName": search_loss_name,
        "ops":        (", ".join(r["search_space_ops"]) if r.get("search_space_ops")
                       else ("+, -, *, /, exp, sqrt, abs, sin, cos, log" if is_pysr else None)),
        "nconst":     r.get("has_const") or ("yes (PySR constants always optimised)" if is_pysr else None),
        "dagSize":    r.get("dag_size"),
        "dagText":    r.get("dag_text"),
        "dagSvg":     r.get("dag_svg"),
        "dagData":    r.get("dag_data") or [],
        "hallOfFame": r.get("hall_of_fame") or [],
        # ── Statistical visualisations (inline SVG) ─────────────────
        "landscapeSvg": r.get("landscape_svg"),     # Dragon: 3-panel landscape
        "paretoSvg":    r.get("pareto_svg"),        # PySR: Pareto frontier
        "treeSvg":      r.get("tree_svg"),          # PySR: sympy/Graphviz AST
        "treeData":     r.get("tree_data"),         # PySR: JSON tree for JS renderer
        "channels":      [
            {
                "tag":  f"ch[{c['idx']}]" + (" *" if c.get('selected') else ""),
                "text": (f"loss={c['loss']:.3e}  \u00b7  {c['formula']}"
                          if c.get('loss') is not None else str(c['formula'])),
            }
            for c in (r.get("all_channel_formulas") or [])
        ],
        "notes":         str(r.get("description", "")) or None,
        # ── Per-method formula variants ─────────────────────────────────
        "formulaChannel":  str(r["formula_channel"]) if r.get("formula_channel") else None,
        "formulaOls":      str(r["formula_ols"]) if r.get("formula_ols") else None,
        "formulaNested":   str(r["formula_nested"]) if r.get("formula_nested") else None,
        "formulaPolyrat":  {str(k): str(v) for k, v in (r.get("formula_polyrat") or {}).items()},
        # ── Per-method losses ────────────────────────────────────────────────
        "lossChannel": float(r["loss_channel"]) if r.get("loss_channel") is not None else None,
        "lossOls":     float(r["loss_ols"])     if r.get("loss_ols")     is not None else None,
        "lossNested":  float(r["loss_nested"])  if r.get("loss_nested")  is not None else None,
        "lossPolyrat": {str(k): float(v) for k, v in (r.get("loss_polyrat") or {}).items()},
        # ── Per-method raw MSE (absolute) ────────────────────────────────────
        "mseCh":      float(r["mse_channel"]) if r.get("mse_channel") is not None else None,
        "mseOls":     float(r["mse_ols"])     if r.get("mse_ols")     is not None else None,
        "mseNested":  float(r["mse_nested"])  if r.get("mse_nested")  is not None else None,
        "msePolyrat": {str(k): float(v) for k, v in (r.get("mse_polyrat") or {}).items()},
    }


def build_html(results, notebook_path="dragonfsr_leaderboard_v2.html", full_leaderboard=True):
    """
    1. Inject PRELOADED_DB into the VS Code notebook template (if it exists).
    2. Write leaderboard_standalone.html using the dragonfsr_leaderboard_v2 design.
    """
    import re as _re

    # ── Build DB dict ────────────────────────────────────────────────────────
    db = {}
    for r in results:
        key      = f"{r['target']}__{r['method']}__{int(r.get('run_id', 0))}"
        db[key]  = _result_to_db_entry(r)
    db_json = json.dumps(db, ensure_ascii=False).replace("</", "<\\/")

    # ── Per-formula aggregate stats (boxplot + heatmap across method × run) ─
    formula_stats = {}
    by_target = {}
    for r in results:
        by_target.setdefault(r["target"], []).append({
            "method": r.get("method"),
            "run":    int(r.get("run_id", 0)),
            "loss":   (float(r["loss"]) if r.get("loss") is not None
                       and np.isfinite(float(r["loss"])) else None),
            "runtime": (float(r["time_s"]) if r.get("time_s") is not None else None),
        })
    for tgt, cells in by_target.items():
        svg = _make_formula_stats_svg(cells)
        if svg:
            formula_stats[tgt] = svg
    formula_stats_json = json.dumps(formula_stats, ensure_ascii=False).replace("</", "<\\/")

    # ── Update notebook template (inject PRELOADED_DB constant) ─────────────
    tmpl = Path(__file__).parent / notebook_path
    if tmpl.exists():
        html = tmpl.read_text(encoding="utf-8")
        new_decl = f"const PRELOADED_DB = {db_json};"
        patched  = _re.sub(
            r"const PRELOADED_DB\s*=\s*\{[^;]*\};",
            new_decl, html, count=1, flags=_re.DOTALL)
        if patched == html:
            # pattern not found — append declaration before first </script>
            patched = html.replace("</script>", f"const PRELOADED_DB = {db_json};\n</script>", 1)
        tmpl.write_text(patched, encoding="utf-8")
        print(f"✓ Template updated ({len(db)} cells): {tmpl}")

    # ── Write standalone HTML ────────────────────────────────────────────────
    _build_standalone_html(db, db_json, formula_stats_json, full_leaderboard)


def _build_standalone_html(db: dict, db_json: str, formula_stats_json: str = "{}", full_leaderboard: bool = True) -> None:
    """
    Write leaderboard_standalone.html using the exact dragonfsr_leaderboard_v2
    design from the attachment.  PRELOADED_DB is injected as a JS constant;
    manual additions in VS Code notebook are merged on top via window.storage.
    """

    # ── CSS variable defaults (standalone browser fallback) ──────────────────
    CSS_VARS = """\
:root{
  --color-text-primary:#1a1a1a;--color-text-secondary:#6b6b6b;
  --color-text-tertiary:#9a9a9a;--color-background-primary:#ffffff;
  --color-background-secondary:#f7f7f7;--color-border-secondary:#e0e0e0;
  --color-border-tertiary:#ebebeb;
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --font-mono:"SF Mono","Fira Code",Consolas,monospace;
}
@media(prefers-color-scheme:dark){
  :root{
    --color-text-primary:#e8e8e8;--color-text-secondary:#a0a0a0;
    --color-text-tertiary:#6a6a6a;--color-background-primary:#1e1e1e;
    --color-background-secondary:#252525;--color-border-secondary:#3d3d3d;
    --color-border-tertiary:#303030;
  }
}"""

    # ── Attachment CSS (verbatim) ─────────────────────────────────────────────
    CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
.root{padding:1rem 0;color:var(--color-text-primary);font-family:var(--font-sans)}
.title{font-size:19px;font-weight:500;letter-spacing:-0.02em;margin-bottom:3px}
.subtitle{font-size:11px;color:var(--color-text-secondary);margin-bottom:1.2rem}
.statbar{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1.2rem}
.sc{background:var(--color-background-secondary);border-radius:8px;padding:9px 11px;text-align:center}
.sc .v{font-size:18px;font-weight:500}
.sc .l{font-size:9px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.tabs{display:flex;gap:5px;margin-bottom:.9rem;flex-wrap:wrap}
.tab{font-size:10px;padding:3px 9px;border:0.5px solid var(--color-border-secondary);border-radius:5px;cursor:pointer;background:var(--color-background-secondary);color:var(--color-text-secondary);transition:all .12s}
.tab.active{background:var(--color-text-primary);color:var(--color-background-primary);border-color:var(--color-text-primary)}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:.9rem;font-size:10px;color:var(--color-text-secondary);align-items:center}
.ld{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:3px}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:1200px;font-size:10px;table-layout:auto}
thead th{font-size:9px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;color:var(--color-text-secondary);padding:5px 7px;text-align:center;border-bottom:1px solid var(--color-border-secondary);white-space:nowrap;background:var(--color-background-primary)}
thead th.fcol{text-align:left;min-width:145px;position:sticky;left:0;z-index:3;background:var(--color-background-primary)}
.gh-pysr{background:#E6F1FB!important;color:#0C447C!important}
.gh-drag{background:#EEEDFE!important;color:#3C3489!important}
tbody td{padding:4px 6px;border-bottom:0.5px solid var(--color-border-tertiary);vertical-align:top;text-align:center}
tbody td.fcol{text-align:left;font-weight:500;font-size:10px;position:sticky;left:0;background:var(--color-background-primary);z-index:1}
tbody td.fcol .ftex{font-size:9px;color:var(--color-text-tertiary);font-weight:400;margin-top:1px;font-style:italic}
tbody tr:hover td{background:var(--color-background-secondary)}
tbody tr:hover td.fcol{background:var(--color-background-secondary)}
.secrow td{background:var(--color-background-secondary)!important;font-size:9px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.07em;padding:4px 7px}
.cg{display:flex;flex-direction:column;gap:2px;align-items:center;min-width:52px}
.runrow{display:flex;gap:2px;justify-content:center}
.pill{display:inline-flex;align-items:center;justify-content:center;height:17px;min-width:22px;padding:0 4px;border-radius:3px;font-size:8px;font-weight:500;cursor:pointer;transition:transform .08s;white-space:nowrap}
.pill:hover{transform:scale(1.08)}
.pill.empty{background:var(--color-background-secondary);color:var(--color-text-tertiary);border:0.5px dashed var(--color-border-tertiary)}
.pill.perfect{background:#EAF3DE;color:#27500A;border:0.5px solid #97C459}
.pill.great{background:#E6F1FB;color:#0C447C;border:0.5px solid #85B7EB}
.pill.ok{background:#FAEEDA;color:#633806;border:0.5px solid #EF9F27}
.pill.fail{background:#FCEBEB;color:#791F1F;border:0.5px solid #F09595}
.smini{font-size:8px;color:var(--color-text-tertiary);text-align:center;line-height:1.5;margin-top:1px}
.winner-badge{font-size:7px;padding:1px 3px;border-radius:2px;font-weight:500}
.wb-polyrat{background:#EEEDFE;color:#3C3489}
.wb-nested{background:#E1F5EE;color:#085041}
.wb-ols{background:#E6F1FB;color:#0C447C}
.wb-ch{background:#F1EFE8;color:#444441}
.modal-wrap{display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.55);align-items:flex-start;justify-content:center;padding:24px 0;overflow-y:auto;backdrop-filter:blur(2px);-webkit-backdrop-filter:blur(2px)}
.modal{position:relative;background:var(--color-background-primary);border:0.5px solid var(--color-border-secondary);border-radius:12px;padding:1.25rem;width:1100px;max-width:98vw;max-height:calc(100vh - 48px);overflow-y:auto;box-shadow:0 18px 48px rgba(0,0,0,0.35)}
.modal-close{position:absolute;top:10px;right:14px;width:30px;height:30px;border-radius:50%;border:0.5px solid var(--color-border-secondary);background:var(--color-background-secondary);color:var(--color-text-primary);cursor:pointer;font-size:18px;line-height:1;display:flex;align-items:center;justify-content:center;font-family:inherit}
.modal-close:hover{background:var(--color-background-tertiary,#eee)}
.modal h3{font-size:14px;font-weight:500;margin-bottom:1rem}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:.8rem}
.mf{display:flex;flex-direction:column;gap:3px}
.mf label{font-size:10px;color:var(--color-text-secondary)}
.mf input,.mf select,.mf textarea{font-size:11px;padding:5px 7px;border:0.5px solid var(--color-border-secondary);border-radius:6px;background:var(--color-background-secondary);color:var(--color-text-primary);width:100%}
.mf textarea{resize:vertical;min-height:50px}
.mf.full{grid-column:1/-1}
.mactions{display:flex;gap:7px;justify-content:flex-end;margin-top:.8rem}
.btn-p{font-size:11px;padding:5px 13px;border-radius:6px;border:0.5px solid var(--color-text-primary);background:var(--color-text-primary);color:var(--color-background-primary);cursor:pointer}
.btn-s{font-size:11px;padding:5px 13px;border-radius:6px;border:0.5px solid var(--color-border-secondary);background:transparent;color:var(--color-text-secondary);cursor:pointer}
.section-divider{border-top:0.5px solid var(--color-border-tertiary);margin:10px 0 6px;padding-top:6px}
.sec-label{font-size:9px;font-weight:500;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px}
.ch-row{display:flex;align-items:center;gap:5px;margin-bottom:4px}
.ch-tag{font-size:9px;background:var(--color-background-secondary);border-radius:3px;padding:1px 4px;color:var(--color-text-secondary);min-width:28px;text-align:center}
.add-ch-btn{font-size:9px;padding:2px 6px;border:0.5px dashed var(--color-border-secondary);border-radius:4px;cursor:pointer;background:transparent;color:var(--color-text-secondary)}
.rm-ch{font-size:9px;cursor:pointer;color:var(--color-text-tertiary);background:transparent;border:none;padding:0 2px}
.ftabs{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
.ftab{font-size:9px;padding:2px 8px;border-radius:12px;border:0.5px solid var(--color-border-secondary);background:transparent;color:var(--color-text-secondary);cursor:pointer;transition:background .15s,color .15s}
.ftab:hover{background:var(--color-background-secondary)}
.ftab.ftab-active{background:var(--color-text-primary);color:var(--color-background-primary);border-color:var(--color-text-primary)}
.ftab.ftab-winner{border-color:#3b6d11;color:#3b6d11}
.ftab.ftab-winner.ftab-active{background:#3b6d11;color:#fff}
.formula-view{font-size:10px;font-family:var(--font-mono);padding:7px 9px;border:0.5px solid var(--color-border-secondary);border-radius:6px;background:var(--color-background-secondary);color:var(--color-text-primary);width:100%;resize:vertical;min-height:60px;box-sizing:border-box}
.tt{background:var(--color-background-primary);border:0.5px solid var(--color-border-secondary);border-radius:8px;padding:9px 11px;font-size:10px;z-index:999;pointer-events:none;min-width:210px;position:absolute;display:none}
.tt .tttitle{font-weight:500;margin-bottom:5px;font-size:11px}
.tt .ttr{display:flex;justify-content:space-between;gap:14px;margin-bottom:2px}
.tt .ttr span:last-child{color:var(--color-text-primary);font-weight:500}
.tt .ttr span:first-child{color:var(--color-text-secondary)}
.phase-pills{display:flex;gap:2px;flex-wrap:wrap;justify-content:center;margin-top:2px}
.pp{font-size:7px;padding:1px 4px;border-radius:2px}
.pp-alg{background:#F1EFE8;color:#444441}
.pp-ln{background:#FAEEDA;color:#633806}
.pp-tri{background:#EEEDFE;color:#3C3489}
.pp-expl{background:#E6F1FB;color:#0C447C}
.fn{margin-top:1.2rem;font-size:9px;color:var(--color-text-tertiary);border-top:0.5px solid var(--color-border-tertiary);padding-top:8px;line-height:1.7}"""

    # ── HTML body (verbatim from attachment) ─────────────────────────────────
    BODY = """\
<div class="root">
"""
    if full_leaderboard:
        formula_defs_js = r"""
const FORMULAS=[
  {id:'hubble',name:'Hubble',cat:'physics',tex:'v = H_0 \\cdot d'},
  {id:'newton',name:'Newton gravity',cat:'physics',tex:'F = G m_1 m_2 / r^2'},
  {id:'rydberg',name:'Rydberg',cat:'physics',tex:'1/\\lambda = R(1/n_1^2 - 1/n_2^2)'},
  {id:'idealgas',name:'Ideal Gas',cat:'physics',tex:'PV = nRT'},
  {id:'kepler',name:"Kepler 3rd",cat:'physics',tex:'T^2 = (4\pi^2/GM)\,a^3'},
  {id:'bode',name:"Bode's law",cat:'physics',tex:'a_n = 0.4 + 0.3 \cdot 2^n'},
  {id:'schechter',name:'Schechter',cat:'physics',tex:'\phi(L)=\phi^*(L/L^*)^\alpha e^{{-L/L^*}}'},
  {id:'leavitt',name:'Leavitt',cat:'physics',tex:'M = a\log P + b'},
  {id:'planck',name:"Planck's law",cat:'physics',tex:'B(\nu,T)=\frac{{2h\nu^3}}{{c^2}}\frac{{1}}{{e^{{h\nu/kT}}-1}}'},
  {id:'n4',name:'Nguyen 4',cat:'nguyen',tex:'x^6+x^5+x^4+x^3+x^2+x'},
  {id:'n5',name:'Nguyen 5',cat:'nguyen',tex:'\sin(x^2)\cos(x)-1'},
  {id:'n6',name:'Nguyen 6',cat:'nguyen',tex:'\sin(x)+\sin(x+x^2)'},
  {id:'n7',name:'Nguyen 7',cat:'nguyen',tex:'\ln(x+1)+\ln(x^2+1)'},
  {id:'n8',name:'Nguyen 8',cat:'nguyen',tex:'\sqrt{{x}}'},
  {id:'n9',name:'Nguyen 9',cat:'nguyen',tex:'\sin(x)+\sin(y^2)'},
  {id:'n10',name:'Nguyen 10',cat:'nguyen',tex:'2\sin(x)\cos(y)'},
  {id:'n11',name:'Nguyen 11',cat:'nguyen',tex:'x^y'},
  {id:'n12',name:'Nguyen 12',cat:'nguyen',tex:'x^4-x^3+y^2/2-y'},
  {id:'ndvi',name:'NDVI',cat:'remote',tex:'\frac{{B_8-B_4}}{{B_8+B_4}}'},
  {id:'wi2015',name:'WI2015',cat:'remote',tex:'1.72+171(B_2+B_3+B_4)-3B_2B_3-1.8B_2B_4-48B_3B_4-0.8B_8B_{{11}}'},
  {id:'awei_sh',name:'AWEI_sh',cat:'remote',tex:'B_2+2.5B_3-1.5(B_{{11}}+B_{{12}})-0.25B_8'},
  {id:'bai',name:'BAI',cat:'remote',tex:'\frac{{1}}{{(0.1-B_4)^2+(0.06-B_8)^2}}'},
  {id:'bsi',name:'BSI',cat:'remote',tex:'\frac{{B_{{11}}+B_4-B_8-B_2}}{{B_{{11}}+B_4+B_8+B_2}}'},
  {id:'evi2',name:'EVI2',cat:'remote',tex:'\frac{{2.5(B_8-B_4)}}{{B_8+2.4B_4+1}}'},
  {id:'vari',name:'VARI',cat:'remote',tex:'\frac{{B_3-B_4}}{{B_3+B_4-B_2}}'},
  {id:'savi',name:'SAVI',cat:'remote',tex:'\frac{{1.5(B_8-B_4)}}{{B_8+B_4+0.5}}'},
  {id:'nirv',name:'NIRv',cat:'remote',tex:'B_8\cdot\frac{{B_8-B_4}}{{B_8+B_4}}'},
];
"""
    else:
        targets=[]
        seen=set()
        for key in db.keys():
            tid = str(key.split('__', 1)[0]).strip()
            if not tid or tid in seen:
                continue
            seen.add(tid)
            targets.append(tid)
        entries = ','.join(
            f"{{id:{json.dumps(t)},name:{json.dumps(t)},cat:'other',tex:''}}" for t in targets
        )
        formula_defs_js = f"const FORMULAS=[{entries}];"

    BODY += """\
<div class="legend">
  <span><span class="ld" style="background:#3b6d11"></span>very low (&lt;1e-6)</span>
  <span><span class="ld" style="background:#185fa5"></span>low (&lt;1e-3)</span>
  <span><span class="ld" style="background:#ba7517"></span>medium (&lt;0.1)</span>
  <span><span class="ld" style="background:#a32d2d"></span>high (≥0.1)</span>
  <span style="margin-left:8px;color:var(--color-text-tertiary)">winner badge:</span>
  <span class="winner-badge wb-polyrat">P-RAT</span>
  <span class="winner-badge wb-nested">NEST</span>
  <span class="winner-badge wb-ols">OLS</span>
  <span class="winner-badge wb-ch">CH</span>
</div>

<div class="scroll">
<table id="tbl">
<thead id="thead"></thead>
<tbody id="tbody"></tbody>
</table>
</div>

<div id="ttbox" class="tt"></div>

<div class="modal-wrap" id="mwrap" style="display:none" onclick="if(event.target===this)closeModal()">
<div class="modal" id="mbox">
  <button type="button" class="modal-close" onclick="closeModal()" title="Close (Esc)" aria-label="Close">×</button>
  <h3 id="mtitle">Log run</h3>
  <div class="mgrid">
    <div class="mf full"><label>Formula · method · run</label><input id="mflabel" readonly style="opacity:.6"/></div>
    <div class="mf full"><label>Init strategy (read-only)</label><input id="minit" readonly style="opacity:.6"/></div>
    <div style="grid-column:1/-1" class="section-divider" id="sec-channels">
      <div class="sec-label" id="sec-channels-label">Output channels — all formulas (one per channel)</div>
      <div id="chlist" style="max-height:260px;overflow-y:auto"></div>
    </div>
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Symbolic formulas <span style="font-weight:400;opacity:.7">(toggle to view each method's output)</span></div>
      <div class="ftabs" id="ftabs"></div>
      <div id="ftab-lossinfo" style="font-size:9px;color:var(--color-text-secondary);margin-bottom:4px;font-family:var(--font-mono)"></div>
      <textarea id="mfexpr" class="formula-view" placeholder="(no formula)" readonly></textarea>
    </div>
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Budget &amp; search space</div>
      <div class="mgrid" style="margin:0">
        <div class="mf"><label>Runtime (s)</label><input id="mrt" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Total iterations (T)</label><input id="miter" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Population K</label><input id="mpop" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Max complexity reached</label><input id="mcomp" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Min search loss reached</label><input id="mphase" readonly style="opacity:.6"/></div>
        <div class="mf"><label>Ops used (list)</label><input id="mops" readonly style="opacity:.6"/></div>
        <div class="mf"><label>n_const (constants optimised)</label><input id="mnconst" readonly style="opacity:.6"/></div>
        <div class="mf"><label>DAG tree size (nodes)</label><input id="mdag" readonly style="opacity:.6"/></div>
      </div>
      <div id="mdagviz" style="margin-top:8px;font-family:var(--font-mono);font-size:10px;white-space:pre;overflow:auto;max-height:520px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:6px;display:none"></div>
    </div>
    <div style="grid-column:1/-1" class="section-divider">
      <div class="sec-label">Search statistics &amp; landscape</div>
      <div id="mlandscape" style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow-x:auto"></div>
      <div id="mpareto"    style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow:auto;max-height:480px"></div>
      <div id="mtree"      style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow:auto;max-height:480px">
        <div style="font-size:10px;color:var(--color-text-secondary);margin-bottom:4px;text-align:left">Best PySR formula — syntactic AST (sympy parse tree)</div>
        <div id="mtree-body"></div>
      </div>
      <div id="mformstats" style="margin-top:6px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-secondary);border-radius:4px;padding:8px;display:none;text-align:center;overflow-x:auto">
        <div style="font-size:10px;color:var(--color-text-secondary);margin-bottom:4px;text-align:left">Aggregated stats for this formula — box-plot per method · ECDF performance profile · loss-vs-runtime trade-off</div>
        <div id="mformstats-body"></div>
      </div>
    </div>
    <div class="mf full" style="margin-top:8px"><label>Notes</label><input id="mnotes" readonly style="opacity:.6"/></div>
  </div>
  <div class="mactions">
    <button class="btn-s" onclick="closeModal()">Close</button>
  </div>
</div>
</div>
"""

    init_strategies = list(getattr(_CfgExp, "INIT_STRATEGIES", []) or [])
    if not init_strategies:
      init_strategies = ["random"]
    n_runs = int(getattr(_CfgExp, "N_RUNS", len(init_strategies)) or len(init_strategies))
    if len(init_strategies) < n_runs:
      init_strategies += [init_strategies[-1]] * (n_runs - len(init_strategies))
    minit_labels = [
      _INIT_STRATEGY_LABELS.get(str(strategy).strip().lower(), str(strategy))
      for strategy in init_strategies[:n_runs]
    ]

    JS = f"""\
const PRELOADED_DB={db_json};
const FORMULA_STATS_SVG={formula_stats_json};
{formula_defs_js}
const METHODS=['pysr','pysr_noise','allops','spar','boosted_spar','allops_const','noolsratn','spar_denoise'];
  const MINIT={json.dumps(minit_labels, ensure_ascii=False)};
const RUN_COUNT=Math.max(1, MINIT.length);
const PHASE_LABELS={{alg:'Algebraic',ln:'+Ln/Exp',trig:'+Sin/Cos',exploit:'Exploit'}};
const PHASE_CLS={{alg:'pp-alg',ln:'pp-ln',trig:'pp-trig',exploit:'pp-expl'}};
const METHOD_LABELS={{pysr:'Baseline',pysr_noise:'+Noise (GP denoising)',allops:'All ops ★ (ref)',spar:'Smart par***',boosted_spar:'Boosted spar***',allops_const:'+ConstBrick',noolsratn:'No OLS/rat/nest',spar_denoise:'Smart par w/ denoise'}};

let FULL_LEADERBOARD={json.dumps(full_leaderboard)};

function visibleMethods(){{
  return METHODS.filter(m=>FULL_LEADERBOARD||Object.keys(DB).some(k=>k.split('__')[1]===m));
}}

function hasDbEntry(fid){{
  if(FULL_LEADERBOARD) return true;
  return Object.keys(DB).some(k=>k.startsWith(`${{fid}}__`));
}}

function buildHeader(){{
  const methods=visibleMethods();
  const thead=document.getElementById('thead');
  thead.innerHTML='';
  const pyMethods=methods.filter(m=>m.startsWith('pysr'));
  const dragMethods=methods.filter(m=>!m.startsWith('pysr'));
  const topRow=document.createElement('tr');
  const fcol=document.createElement('th');
  fcol.className='fcol';
  fcol.rowSpan=2;
  fcol.style='vertical-align:bottom';
  fcol.textContent='Formula';
  topRow.appendChild(fcol);
  if(pyMethods.length){{
    const th=document.createElement('th');
    th.className='gh-pysr';
    th.colSpan=pyMethods.length;
    th.textContent='PySR';
    topRow.appendChild(th);
  }}
  if(dragMethods.length){{
    const th=document.createElement('th');
    th.className='gh-drag';
    th.colSpan=dragMethods.length;
    th.textContent='DragonSR';
    topRow.appendChild(th);
  }}
  thead.appendChild(topRow);
  const secondRow=document.createElement('tr');
  methods.forEach(m=>{{
    const th=document.createElement('th');
    th.className=m.startsWith('pysr')?'gh-pysr':'gh-drag';
    th.textContent=METHOD_LABELS[m]||m;
    secondRow.appendChild(th);
  }});
  thead.appendChild(secondRow);
}}

let DB={{}};
let activeCat='all';
let pending=null;
let chCount=0;

function key(f,m,r){{return `${{f}}__${{m}}__${{r}}`}}

function lossClass(v){{
  if(v===null||v===undefined) return 'empty';
  if(v<1e-6) return 'perfect';
  if(v<1e-3) return 'great';
  if(v<0.1) return 'ok';
  return 'fail';
}}
function fmtLoss(v){{
  if(v===null||v===undefined) return '—';
  if(v<0.001) return v.toExponential(2);
  return v.toFixed(4);
}}
function winnerBadge(w){{
  if(!w) return '';
  const map={{polyrat:'wb-polyrat',nested:'wb-nested',ols:'wb-ols',channel:'wb-ch'}};
  const lbl={{polyrat:'P-RAT',nested:'NEST',ols:'OLS',channel:'CH'}};
  return `<span class="winner-badge ${{map[w]||''}}">${{lbl[w]||w}}</span>`;
}}
function phasePill(ph){{
  if(!ph) return '';
  return `<span class="pp ${{PHASE_CLS[ph]||''}}">${{PHASE_LABELS[ph]||ph}}</span>`;
}}

function initLabel(run, mid){{
  if(mid==='pysr' || mid==='pysr_noise') return String(run+1);
  return `R${{run+1}}`;
}}

function initStrategyText(run){{
  return MINIT[run] || MINIT[MINIT.length-1] || `R${{run+1}}`;
}}

function renderCell(fid,mid){{
  let html='<div class="cg"><div class="runrow">';
  let vals=[],rts=[],polyrats=0;
  for(let r=0;r<RUN_COUNT;r++){{
    const k=key(fid,mid,r);
    const d=DB[k];
    if(!d || (!FULL_LEADERBOARD && (d.oneMinusR2===null||d.oneMinusR2===undefined))){{
      if(FULL_LEADERBOARD){{
        const lbl=initLabel(r, mid);
        html+=`<div class="pill empty" data-fid="${{fid}}" data-mid="${{mid}}" data-run="${{r}}"
          onmouseenter="showTT(event,'${{fid}}','${{mid}}',${{r}})"
          onmouseleave="hideTT()"
          onclick="openModal('${{fid}}','${{mid}}',${{r}})">${{lbl}}</div>`;
      }}
      continue;
    }}
    const v=d.oneMinusR2;
    const cls=lossClass(v);
    const lbl=v!==null&&v!==undefined?fmtLoss(v):initLabel(r, mid);
    html+=`<div class="pill ${{cls}}" data-fid="${{fid}}" data-mid="${{mid}}" data-run="${{r}}"
      onmouseenter="showTT(event,'${{fid}}','${{mid}}',${{r}})"
      onmouseleave="hideTT()"
      onclick="openModal('${{fid}}','${{mid}}',${{r}})">${{lbl}}</div>`;
    if(v!==null){{vals.push(v);if(d.runtime)rts.push(d.runtime);if(d.winner==='polyrat')polyrats++;}}
  }}
  html+='</div>';
  if(vals.length>=2){{
    const mu=vals.reduce((a,b)=>a+b,0)/vals.length;
    const sig=Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
    html+=`<div class="smini">μ=${{fmtLoss(mu)}} σ=${{sig.toExponential(1)}}</div>`;
  }}
  const runIdx=[...Array(RUN_COUNT).keys()];
  const winners=runIdx.map(r=>DB[key(fid,mid,r)]?.winner);
  const anyWin=winners.some(Boolean);
  if(anyWin){{
    const badges=winners.map((w,i)=>{{
      if(!w) return `<span class="winner-badge" style="opacity:.25" title="Run ${{i+1}}: —">R${{i+1}}</span>`;
      const b=winnerBadge(w);
      return b.replace('<span class="winner-badge', `<span title="Run ${{i+1}}: ${{w}}" class="winner-badge`);
    }}).join(' ');
    html+=`<div style="margin-top:2px;display:flex;gap:2px;flex-wrap:wrap">${{badges}}</div>`;
  }}
  const phases=runIdx.map(r=>DB[key(fid,mid,r)]?.phase).filter(Boolean);
  if(phases.length){{
    html+=`<div class="phase-pills">${{[...new Set(phases)].map(p=>phasePill(p)).join('')}}</div>`;
  }}
  // Per-run runtimes (R1..RN) listed individually
  const parts=runIdx.reduce((acc,r)=>{{const dd=DB[key(fid,mid,r)]; if(dd!=null && dd.runtime!=null){{ acc.push(`R${{r+1}}:\u202f${{dd.runtime.toFixed(1)}}s`);}} return acc; }}, []);
  if(parts.length){{
    html+=`<div class="smini" style="white-space:normal;line-height:1.3">${{parts.join(' · ')}}</div>`;
  }}
  html+='</div>';
  return html;
}}

function buildTable(){{
  buildHeader();
  const tbody=document.getElementById('tbody');
  tbody.innerHTML='';
  const methods=visibleMethods();
  if(!FULL_LEADERBOARD){{
    const rows=FORMULAS.filter(f=>hasDbEntry(f.id));
    rows.forEach(f=>{{
      const tr=document.createElement('tr');
      let cells=`<td class="fcol"><div>${{f.name}}</div></td>`;
      methods.forEach(m=>{{cells+=`<td id="cell_${{f.id}}_${{m}}">${{renderCell(f.id,m)}}</td>`;}});
      tr.innerHTML=cells;tbody.appendChild(tr);
    }});
  }} else {{
    const cats=['physics','nguyen','remote','other'];
    const cnames={{physics:'Physics laws',nguyen:'Nguyen benchmark (1–12)',remote:'Remote sensing indices',other:'Other'}};
    cats.forEach(cat=>{{
      const rows=FORMULAS.filter(f=>f.cat===cat&&(activeCat==='all'||activeCat===cat)&&hasDbEntry(f.id));
      if(!rows.length) return;
      const sr=document.createElement('tr');sr.className='secrow';
      sr.innerHTML=`<td class="fcol">${{cnames[cat]}}</td>${{methods.map(()=>'<td></td>').join('')}}`;
      tbody.appendChild(sr);
      rows.forEach(f=>{{
        const tr=document.createElement('tr');
        let cells=`<td class="fcol"><div>${{f.name}}</div>${{f.tex?`<div class="ftex">$${{f.tex}}$</div>`:''}}</td>`;
        methods.forEach(m=>{{cells+=`<td id="cell_${{f.id}}_${{m}}">${{renderCell(f.id,m)}}</td>`;}});
        tr.innerHTML=cells;tbody.appendChild(tr);
      }});
    }});
  }}
  updateStats();
  // Render LaTeX in the formula column (KaTeX auto-render).
  if(window.renderMathInElement){{
    try{{
      renderMathInElement(document.getElementById('tbl'),{{
        delimiters:[
          {{left:'$$',right:'$$',display:true}},
          {{left:'$',right:'$',display:false}},
        ],
        throwOnError:false,
      }});
    }}catch(e){{}}
  }}
}}

function updateStats(){{
  let runs=0,fills=new Set(),best=null,polyrats=0,rts=[];
  Object.entries(DB).forEach(([k,d])=>{{
    if(!d) return;
    if(d.oneMinusR2!==null&&d.oneMinusR2!==undefined){{
      runs++;
      const fid=k.split('__')[0];fills.add(fid);
      if(best===null||d.oneMinusR2<best) best=d.oneMinusR2;
      if(d.winner==='polyrat') polyrats++;
    }}
    if(d.runtime) rts.push(d.runtime);
  }});
  document.getElementById('s-runs').textContent=runs;
  document.getElementById('s-fills').textContent=`${{fills.size}}/${{FORMULAS.length}}`;
  document.getElementById('s-best').textContent=best!==null?fmtLoss(best):'—';
  document.getElementById('s-polyrat').textContent=polyrats;
  document.getElementById('s-rt').textContent=rts.length?(rts.reduce((a,b)=>a+b,0)/rts.length).toFixed(1)+'s':'—';
}}

function _renderChannelsRO(channels){{
  const div=document.getElementById('chlist');
  div.innerHTML='';
  if(!channels||!channels.length){{
    div.innerHTML='<div style="font-size:10px;opacity:.6;padding:4px">(no channel info)</div>';
    return;
  }}
  channels.forEach((c,i)=>{{
    const row=document.createElement('div');row.className='ch-row';
    row.style.cssText='display:flex;align-items:flex-start;gap:6px;padding:3px 4px;border-bottom:0.5px solid var(--color-border-tertiary)';
    row.innerHTML=`<span class="ch-tag" style="font-family:var(--font-mono);font-size:10px;min-width:60px;color:var(--color-text-secondary)">${{c.tag||('ch['+i+']')}}</span>
      <span style="flex:1;font-family:var(--font-mono);font-size:10px;white-space:pre-wrap;word-break:break-all;color:var(--color-text-primary)">${{(c.text||c).toString().replace(/</g,'&lt;')}}</span>`;
    div.appendChild(row);
  }});
}}
function _renderHallOfFame(hof){{
  const div=document.getElementById('chlist');
  div.innerHTML='';
  if(!hof||!hof.length){{
    div.innerHTML='<div style="font-size:10px;opacity:.6;padding:4px">(no Pareto frontier available)</div>';
    return;
  }}
  const tbl=document.createElement('table');
  tbl.style.cssText='width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:10px';
  tbl.innerHTML=`<thead><tr style="text-align:left;border-bottom:0.5px solid var(--color-border-secondary)">
    <th style="padding:3px 6px;width:30px">#</th>
    <th style="padding:3px 6px;width:60px">cmplx</th>
    <th style="padding:3px 6px;width:110px">loss</th>
    <th style="padding:3px 6px;width:90px" title="PySR parsimony score = -Δln(loss)/Δcomplexity (higher = better gain per added complexity unit)">score</th>
    <th style="padding:3px 6px">expression</th></tr></thead>`;
  const tb=document.createElement('tbody');
  hof.forEach((m,i)=>{{
    const tr=document.createElement('tr');
    tr.style.borderBottom='0.5px solid var(--color-border-tertiary)';
    const lossStr=(m.loss!==null&&m.loss!==undefined)?Number(m.loss).toExponential(3):'—';
    const scoreStr=(m.score!==null&&m.score!==undefined&&isFinite(m.score))?Number(m.score).toFixed(3):'—';
    tr.innerHTML=`<td style="padding:3px 6px;color:var(--color-text-secondary)">${{i+1}}</td>
      <td style="padding:3px 6px">${{m.complexity??'—'}}</td>
      <td style="padding:3px 6px">${{lossStr}}</td>
      <td style="padding:3px 6px;color:var(--color-text-secondary)">${{scoreStr}}</td>
      <td style="padding:3px 6px;white-space:pre-wrap;word-break:break-all">${{(m.formula||'').replace(/</g,'&lt;')}}</td>`;
    tb.appendChild(tr);
  }});
  tbl.appendChild(tb);
  div.appendChild(tbl);
}}

function _buildFormulaTabs(d, isPysr){{
  const container=document.getElementById('ftabs');
  const ta=document.getElementById('mfexpr');
  container.innerHTML='';

  function lossBadge(v){{
    if(v===null||v===undefined) return '';
    let col='#ba7517';
    if(v<1e-6) col='#3b6d11';
    else if(v<1e-3) col='#185fa5';
    else if(v>=0.1) col='#a32d2d';
    return ` <span style="font-size:8px;opacity:.85;color:${{col}}">${{v.toExponential(3)}}</span>`;
  }}

  // Collect all available method formulas (no synthetic "Winner" tab)
  const tabs=[];
  const isCorrMetric=(String(d?.searchLossName||'corr').toLowerCase()==='corr'||
                      String(d?.searchLossName||'corr').toLowerCase()==='channel');
  if(d&&d.formulaChannel) tabs.push({{id:'ch',   label:'Channel',    expr:d.formulaChannel, loss:d.lossChannel, mse:d.mseCh}});
  if(d&&d.formulaOls)     tabs.push({{id:'ols',  label:'Sparse OLS', expr:d.formulaOls,     loss:d.lossOls,     mse:d.mseOls}});
  if(d&&d.formulaNested)  tabs.push({{id:'nest', label:'Nested OLS', expr:d.formulaNested,  loss:d.lossNested,  mse:d.mseNested}});
  if(d&&d.formulaPolyrat){{
    Object.keys(d.formulaPolyrat).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(deg=>{{
      const expr=d.formulaPolyrat[deg];
      const lv=(d.lossPolyrat||{{}})[deg]??null;
      const mv=(d.msePolyrat||{{}})[deg]??null;
      if(expr) tabs.push({{id:'rat'+deg, label:'P-RAT deg\u202f'+deg, expr, loss:lv, mse:mv}});
    }});
  }}

  // Hide the lossInfo line for PySR up-front (no R²/MSE applicable; hidden
  // even on early-return below when there are no per-method tabs).
  const lossInfo=document.getElementById('ftab-lossinfo');
  if(isPysr&&lossInfo){{ lossInfo.textContent=''; lossInfo.style.display='none'; }}
  else if(lossInfo){{ lossInfo.style.display=''; lossInfo.textContent=''; }}

  // Fallback: if no per-method formulas, show the single finalExpr
  if(!tabs.length){{
    ta.value=d?.finalExpr??'(no formula)';
    return;
  }}

  // Auto-select the tab with the lowest loss (= best method)
  let activeIdx=0;
  let bestLoss=Infinity;
  tabs.forEach((tab,i)=>{{
    if(tab.loss!==null&&tab.loss!==undefined&&tab.loss<bestLoss){{
      bestLoss=tab.loss; activeIdx=i;
    }}
  }});

  function activate(idx){{
    activeIdx=idx;
    container.querySelectorAll('.ftab').forEach((b,i)=>{{
      b.classList.toggle('ftab-active',i===idx);
      b.classList.toggle('ftab-winner',i===idx&&tabs[i].loss===bestLoss);
    }});
    ta.value=tabs[idx].expr||'(empty)';
    if(lossInfo){{
      const lv=tabs[idx].loss;
      const mv=tabs[idx].mse;
      const parts=[];
      if(lv!==null&&lv!==undefined) parts.push((isCorrMetric?'1\u2212R\u00b2':'loss')+'\u202f=\u202f'+lv.toExponential(6));
      if(mv!==null&&mv!==undefined) parts.push('MSE\u202f=\u202f'+mv.toExponential(6));
      lossInfo.textContent=parts.join('\u2003');
    }}
  }}

  tabs.forEach((tab,i)=>{{
    const btn=document.createElement('button');
    btn.className='ftab';
    btn.innerHTML=tab.label+lossBadge(tab.loss);
    btn.onclick=()=>activate(i);
    container.appendChild(btn);
  }});
  activate(activeIdx);
}}

// ── JS-based DAG renderer (fallback when graphviz 'dot' is unavailable) ──────
function _renderDagSvg(dagData){{
  const n=(dagData||[]).length;
  if(!n) return null;
  // Build parents list from children
  const parents=Array.from({{length:n}},()=>[]);
  dagData.forEach((node,i)=>{{
    (node.children||[]).forEach(j=>{{ if(j>=0&&j<n) parents[j].push(i); }});
  }});
  // Depth = longest path from root (DP on topological order 0..n-1)
  const depth=new Array(n).fill(0);
  for(let i=1;i<n;i++){{
    if(parents[i].length) depth[i]=Math.max(...parents[i].map(p=>depth[p]))+1;
  }}
  const maxD=Math.max(...depth);
  const layers=Array.from({{length:maxD+1}},()=>[]);
  depth.forEach((d,i)=>layers[d].push(i));
  // Layout constants
  const NW=240,NH=54,HGAP=18,VGAP=40,PAD=20;
  const layerW=layers.map(l=>l.length*(NW+HGAP)-HGAP);
  const totalW=Math.max(...layerW)+2*PAD;
  const totalH=(maxD+1)*(NH+VGAP)-VGAP+2*PAD;
  // Node positions (top-left corner)
  const pos=new Array(n);
  layers.forEach((layer,d)=>{{
    const lw=layer.length*(NW+HGAP)-HGAP;
    const sx=(totalW-lw)/2;
    layer.forEach((nid,k)=>{{ pos[nid]={{x:sx+k*(NW+HGAP),y:PAD+d*(NH+VGAP)}}; }});
  }});
  const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let p=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{totalW}} ${{totalH}}" width="${{totalW}}" height="${{totalH}}" style="max-width:100%;height:auto;background:transparent">`,
    `<defs><marker id="dagar" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon points="0,0 8,3 0,6" fill="#777"/></marker></defs>`];
  // Edges
  dagData.forEach((node,i)=>{{
    (node.children||[]).forEach(j=>{{
      if(j<0||j>=n) return;
      const x1=pos[i].x+NW/2,y1=pos[i].y+NH;
      const x2=pos[j].x+NW/2,y2=pos[j].y;
      const ym=(y1+y2)/2;
      p.push(`<path d="M${{x1}},${{y1}} C${{x1}},${{ym}} ${{x2}},${{ym}} ${{x2}},${{y2}}" fill="none" stroke="#888" stroke-width="1.5" marker-end="url(#dagar)"/>`);
    }});
  }});
  // Nodes — two-line label: line1=[idx] ClassName, line2=hyperparameters
  dagData.forEach((node,i)=>{{
    const {{x,y}}=pos[i];
    const isRoot=(i===0),isLeaf=!(node.children||[]).length;
    const fill=isRoot?'#3b6d11':isLeaf?'#185fa5':'#ffa600';
    const fc=(isRoot||isLeaf)?'#ececec':'#1a1a1a';
    const fullDesc=`[${{i}}] ${{node.desc||''}}`;
    const parts=fullDesc.split(' | ');
    const line1=esc(parts[0].slice(0,34));
    const line2=parts.length>1?esc(parts.slice(1).join(' | ').slice(0,38)):'';
    const ty1=y+(line2?NH/2-5:NH/2+4);
    p.push(
      `<rect x="${{x}}" y="${{y}}" width="${{NW}}" height="${{NH}}" rx="5" fill="${{fill}}" stroke="#444" stroke-width="1"><title>${{esc(fullDesc)}}</title></rect>`,
      `<text text-anchor="middle" font-size="10" font-family="monospace" fill="${{fc}}">`,
      `<tspan x="${{x+NW/2}}" y="${{ty1}}">${{line1}}</tspan>`,
      line2?`<tspan x="${{x+NW/2}}" dy="13">${{line2}}</tspan>`:'',
      `</text>`
    );
  }});
  p.push('</svg>');
  return p.join('');
}}

function _renderFormulaTree(nodes){{
  /* Render a sympy AST as an SVG tree.
     nodes = [{{id, label, kind:"op"|"var"|"const", children:[idx,...]}}]
     Root = node not referenced as any other node's child (auto-detected). */
  if(!nodes||!nodes.length) return null;
  // Auto-detect root (the binarizer appends root last, not at index 0)
  const _allCh=new Set(nodes.flatMap(nd=>(nd.children||[])));
  const root=Math.max(0,nodes.findIndex((_,i)=>!_allCh.has(i)));
  const NW=66,NH=34,HGAP=8,VGAP=38,PAD=18;
  // Compute subtree pixel widths
  const w=new Array(nodes.length).fill(NW);
  function calcW(i){{
    const ch=nodes[i].children||[];
    if(!ch.length){{w[i]=NW;return;}}
    ch.forEach(c=>calcW(c));
    w[i]=Math.max(NW,ch.reduce((s,c)=>s+w[c],0)+Math.max(0,ch.length-1)*HGAP);
  }}
  calcW(root);
  // Assign center-x and top-y for each node
  const cx=new Array(nodes.length).fill(0),cy=new Array(nodes.length).fill(0);
  function layout(i,left,depth){{
    cy[i]=PAD+depth*(NH+VGAP);
    const ch=nodes[i].children||[];
    if(!ch.length){{cx[i]=left+NW/2;return;}}
    let x=left;
    ch.forEach(c=>{{layout(c,x,depth+1);x+=w[c]+HGAP;}});
    cx[i]=(cx[ch[0]]+cx[ch[ch.length-1]])/2;
  }}
  layout(root,PAD,0);
  const totalW=w[root]+2*PAD;
  const totalH=Math.max(...cy)+NH+PAD;
  const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const parts=[
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${{totalW}} ${{totalH}}" width="${{totalW}}" height="${{totalH}}" style="max-width:100%;height:auto;background:transparent">`,
    `<defs><marker id="tarr" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto"><polygon points="0,0 6,2.5 0,5" fill="#888"/></marker></defs>`
  ];
  // Edges (curved bezier)
  nodes.forEach((nd,i)=>{{
    (nd.children||[]).forEach(j=>{{
      const x1=cx[i],y1=cy[i]+NH,x2=cx[j],y2=cy[j],ym=(y1+y2)/2;
      parts.push(`<path d="M${{x1}},${{y1}} C${{x1}},${{ym}} ${{x2}},${{ym}} ${{x2}},${{y2}}" fill="none" stroke="#999" stroke-width="1.4" marker-end="url(#tarr)"/>`);
    }});
  }});
  // Nodes
  nodes.forEach((nd,i)=>{{
    const x=cx[i]-NW/2,y=cy[i];
    const fill=nd.kind==='op'?'#ffa600':nd.kind==='const'?'#185fa5':'#3b6d11';
    const fc=nd.kind==='op'?'#1a1a1a':'#ececec';
    const lbl=esc(String(nd.label||'').slice(0,14));
    parts.push(
      `<rect x="${{x}}" y="${{y}}" width="${{NW}}" height="${{NH}}" rx="5" fill="${{fill}}" stroke="#444" stroke-width="1"/>`,
      `<text x="${{cx[i]}}" y="${{y+NH/2+4}}" text-anchor="middle" font-size="11" font-family="monospace" fill="${{fc}}">${{lbl}}</text>`
    );
  }});
  parts.push('</svg>');
  return parts.join('');
}}

function openModal(fid,mid,run){{
  const f=FORMULAS.find(x=>x.id===fid);
  const midx=METHODS.indexOf(mid);
  const _isPysr = mid==='pysr' || mid==='pysr_noise';
  const mnames=['PySR baseline','PySR +Noise','All ops ★ (ref)','Smart parallel***','Boosted spar***','+ConstantBrick','No OLS/rat/nest','Smart par w/ denoise***'];
  pending={{fid,mid,run}};
  document.getElementById('mtitle').textContent='Log run result';
  document.getElementById('mflabel').value=`${{f.name}}  ·  ${{mnames[midx]}}  ·  Run ${{run+1}}`;
  if(_isPysr){{
    const pysrLbl = mid==='pysr_noise'
      ? `Run ${{run+1}} — PySR +Noise (GP denoising pre-step, denoise=True)`
      : `Run ${{run+1}} — PySR baseline (single deterministic config; only the data seed varies between runs for synthetic targets)`;
    document.getElementById('minit').value=pysrLbl;
  }} else {{
    document.getElementById('minit').value=`R${{run+1}}: ${{initStrategyText(run)}}`;
  }}
  const k=key(fid,mid,run);const d=DB[k]||{{}};
  // ── Formula tabs (suppress R²/MSE info-line for PySR which uses raw MSE) ──
  _buildFormulaTabs(d, _isPysr);
  document.getElementById('mrt').value=d.runtime??'';
  // Total iterations (T): for smart-parallel ('spar') show
  // total = sum(streams)  +  per-stream breakdown  +  winner count.
  {{
    let tval = (d.totalT==null) ? '' : Number(d.totalT).toLocaleString();
    if (d.actualTPerStream && Object.keys(d.actualTPerStream).length){{
      const parts = Object.entries(d.actualTPerStream)
        .map(([k,v])=>`${{k}}=${{Number(v).toLocaleString()}}`).join(', ');
      const winStr = (d.actualTWinner!=null)
        ? `  ·  winner=${{Number(d.actualTWinner).toLocaleString()}}` : '';
      tval = `${{Number(d.totalT).toLocaleString()}}  (Σ streams: ${{parts}}${{winStr}})`;
    }}
    document.getElementById('miter').value = tval;
  }}
  document.getElementById('mpop').value=d.K??'';
  document.getElementById('mcomp').value=d.maxComp??'';
  // Min search loss reached (Dragon: alignment loss = 1−corr; PySR: best MSE)
  {{
    const sl=d.searchLoss;
    document.getElementById('mphase').value=(sl===null||sl===undefined)?'':
      (Number(sl).toExponential(4)+(d.searchLossKind?'  ('+d.searchLossKind+')':''));
  }}
  document.getElementById('mops').value=d.ops??'';
  document.getElementById('mnconst').value=d.nconst??'';
  document.getElementById('mdag').value=d.dagSize??'';
  document.getElementById('mnotes').value=d.notes??'';
  // ── Section: channels (DRAGON) or hall-of-fame (PySR) ──────────────
  const secLbl=document.getElementById('sec-channels-label');
  if(_isPysr){{
    if(secLbl) secLbl.textContent='PySR hall of fame — Pareto frontier (complexity vs loss)';
    _renderHallOfFame(d.hallOfFame||[]);
  }} else {{
    if(secLbl) secLbl.textContent='Output channels — all formulas (one per channel, read-only)';
    _renderChannelsRO(d.channels||[]);
  }}
  // ── DAG visualisation (SVG if available, otherwise textual table) ──
  const dviz=document.getElementById('mdagviz');
  if(dviz){{
    if(d.dagSvg){{
      dviz.style.display='block';
      dviz.style.whiteSpace='normal';
      dviz.style.textAlign='center';
      dviz.innerHTML=d.dagSvg;
      const svgEl=dviz.querySelector('svg');
      if(svgEl){{ svgEl.style.maxWidth='100%'; svgEl.style.height='auto'; }}
    }} else if(d.dagData&&d.dagData.length){{
      // JS-based SVG fallback (graphviz 'dot' not in PATH on this machine)
      const jsSvg=_renderDagSvg(d.dagData);
      if(jsSvg){{
        dviz.style.display='block';
        dviz.style.whiteSpace='normal';
        dviz.style.textAlign='center';
        dviz.innerHTML=jsSvg;
        const svgEl=dviz.querySelector('svg');
        if(svgEl){{ svgEl.style.maxWidth='100%'; svgEl.style.height='auto'; }}
      }} else {{
        dviz.style.display='block';
        dviz.style.whiteSpace='pre';
        dviz.style.textAlign='left';
        dviz.textContent=d.dagText||'';
      }}
    }} else if(d.dagText){{
      dviz.style.display='block';
      dviz.style.whiteSpace='pre';
      dviz.style.textAlign='left';
      dviz.textContent=d.dagText;
    }} else {{
      dviz.style.display='none';
      dviz.innerHTML='';
    }}
  }}
  // ── Statistical visualisations (per-run + per-formula) ─────────────
  function _execScripts(container){{
    // <script> tags inserted via innerHTML do not auto-execute. Re-create them
    // so embedded Plotly.newPlot() calls actually run.
    container.querySelectorAll('script').forEach(old=>{{
      const s=document.createElement('script');
      if(old.src) s.src=old.src; else s.textContent=old.textContent;
      old.parentNode.replaceChild(s,old);
    }});
  }}
  function _setSvgBlock(elId, svg, isWrappedBody){{
    const el=document.getElementById(elId);
    if(!el) return;
    if(svg){{
      el.style.display='block';
      const tgt=isWrappedBody?el.querySelector('#'+elId+'-body'):el;
      if(tgt){{
        tgt.innerHTML=svg;
        const s=tgt.querySelector('svg');
        if(s){{s.style.maxWidth='100%';s.style.height='auto';}}
        _execScripts(tgt);
      }}
    }} else {{
      el.style.display='none';
      const tgt=isWrappedBody?el.querySelector('#'+elId+'-body'):el;
      if(tgt) tgt.innerHTML='';
    }}
  }}
  // IMPORTANT: show the modal BEFORE injecting Plotly content. Plotly reads
  // the container's clientWidth at render time; if the modal is still
  // display:none the width is 0 and the chart is rendered squashed.
  const mwrap=document.getElementById('mwrap');
  mwrap.style.display='flex';
  document.body.style.overflow='hidden';
  // Dragon: 3-panel landscape; PySR: Pareto + AST tree.
  _setSvgBlock('mlandscape', _isPysr ? null : (d.landscapeSvg||null), false);
  _setSvgBlock('mpareto',    _isPysr ? (d.paretoSvg||null) : null,   false);
  if(_isPysr){{
    let _treeSvgContent=d.treeSvg||null;
    if(!_treeSvgContent&&d.treeData&&d.treeData.length)
      _treeSvgContent=_renderFormulaTree(d.treeData)||null;
    _setSvgBlock('mtree',_treeSvgContent,true);
  }}else{{
    _setSvgBlock('mtree',null,true);
  }}
  // Per-formula aggregate stats (boxplot + heatmap), shared across runs/methods.
  _setSvgBlock('mformstats', (typeof FORMULA_STATS_SVG!=='undefined' && FORMULA_STATS_SVG[fid]) || null, true);
  // Force Plotly to re-fit each visible chart now that the modal width is set.
  setTimeout(()=>{{
    if(window.Plotly){{
      ['mlandscape','mpareto','mformstats'].forEach(id=>{{
        const el=document.getElementById(id);
        if(!el||el.style.display==='none') return;
        el.querySelectorAll('.js-plotly-plot').forEach(p=>{{
          try{{ window.Plotly.Plots.resize(p); }}catch(e){{}}
        }});
      }});
    }}
  }},80);
  setTimeout(()=>{{ const mb=document.getElementById('mbox'); if(mb) mb.scrollTop=0; }},20);
}}
function closeModal(){{
  document.getElementById('mwrap').style.display='none';
  document.body.style.overflow='';
  pending=null;
}}
// Esc closes the modal popup
document.addEventListener('keydown',function(e){{
  if(e.key==='Escape'){{
    const mw=document.getElementById('mwrap');
    if(mw&&mw.style.display!=='none') closeModal();
  }}
}});
function saveRun(){{
  if(!pending) return;
  const {{fid,mid,run}}=pending;
  const k=key(fid,mid,run);
  DB[k]={{
    oneMinusR2:DB[k]?.oneMinusR2??null,
    mse:DB[k]?.mse??null,
    searchLoss:DB[k]?.searchLoss??null,
    searchLossKind:DB[k]?.searchLossKind??null,
    searchLossName:DB[k]?.searchLossName??null,
    winner:DB[k]?.winner??null,
    bestCh:DB[k]?.bestCh??null,
    polyDeg:DB[k]?.polyDeg??null,
    nestedLink:DB[k]?.nestedLink??null,
    finalExpr:DB[k]?.finalExpr??null,
    runtime:DB[k]?.runtime??null,
    totalT:DB[k]?.totalT??null,
    actualTWinner:DB[k]?.actualTWinner??null,
    actualTPerStream:DB[k]?.actualTPerStream??{{}},
    K:DB[k]?.K??null,
    maxComp:DB[k]?.maxComp??null,
    phase:DB[k]?.phase??null,
    ops:DB[k]?.ops??null,
    nconst:DB[k]?.nconst??null,
    dagSize:DB[k]?.dagSize??null,
    channels:DB[k]?.channels||[],
    notes:DB[k]?.notes??null,
  }};
  const cell=document.getElementById(`cell_${{fid}}_${{mid}}`);
  if(cell) cell.innerHTML=renderCell(fid,mid);
  closeModal();updateStats();
  try{{window.storage.set('lb_v2',JSON.stringify(DB),false);}}catch(e){{}}
}}

function showTT(e,fid,mid,run){{
  const k=key(fid,mid,run);const d=DB[k];
  const f=FORMULAS.find(x=>x.id===fid);
  const mnames=['PySR','PySR +Noise','All ops ★ (ref)','Smart par','Boosted spar','ConstBrick','No OLS','Smart par+denoise'];
  const midx=METHODS.indexOf(mid);
  const _isCorrMetric=(String(d?.searchLossName||'corr').toLowerCase()==='corr'||
                       String(d?.searchLossName||'corr').toLowerCase()==='channel');
  const _metricLabel=_isCorrMetric?'1−R²':'loss';
  const tt=document.getElementById('ttbox');
  let h=`<div class="tttitle">${{f.name}} · ${{mnames[midx]}} · R${{run+1}}</div>`;
  h+=`<div class="ttr"><span>Init</span><span>${{initStrategyText(run)}}</span></div>`;
  if(d){{
    if(d.oneMinusR2!==null&&d.oneMinusR2!==undefined) h+=`<div class="ttr"><span>${{_metricLabel}}</span><span>${{fmtLoss(d.oneMinusR2)}}</span></div>`;
    if(d.mse!==null&&d.mse!==undefined) h+=`<div class="ttr"><span>MSE</span><span>${{d.mse.toExponential(3)}}</span></div>`;
    if(d.winner) h+=`<div class="ttr"><span>Winner</span><span>${{d.winner}}</span></div>`;
    if(d.polyDeg) h+=`<div class="ttr"><span>P-RAT degree</span><span>≤${{d.polyDeg}}</span></div>`;
    if(d.nestedLink) h+=`<div class="ttr"><span>Nested link</span><span>${{d.nestedLink}}</span></div>`;
    // Per-method score breakdown
    function ttloss(v){{ return (v!==null&&v!==undefined)?v.toExponential(3):'—'; }}
    if(d.lossChannel!==undefined||d.lossOls!==undefined||d.lossNested!==undefined){{
      h+=`<div style="border-top:0.5px solid var(--color-border-tertiary);margin:4px 0 3px"></div>`;
      h+=`<div style="font-size:8px;font-weight:500;color:var(--color-text-secondary);margin-bottom:2px">${{_metricLabel}} per method</div>`;
      if(d.lossChannel!==null&&d.lossChannel!==undefined) h+=`<div class="ttr"><span>Channel</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossChannel)}}</span></div>`;
      if(d.lossOls!==null&&d.lossOls!==undefined)         h+=`<div class="ttr"><span>Sparse OLS</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossOls)}}</span></div>`;
      if(d.lossNested!==null&&d.lossNested!==undefined)   h+=`<div class="ttr"><span>Nested OLS</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossNested)}}</span></div>`;
      if(d.lossPolyrat){{
        Object.keys(d.lossPolyrat).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(deg=>{{
          h+=`<div class="ttr"><span>P-RAT deg ${{deg}}</span><span style="font-family:var(--font-mono)">${{ttloss(d.lossPolyrat[deg])}}</span></div>`;
        }});
      }}
    }}
    if(d.channels&&d.channels.length) h+=`<div class="ttr"><span>Channels</span><span>${{d.channels.length}}</span></div>`;
    if(d.phase) h+=`<div class="ttr"><span>Phase</span><span>${{PHASE_LABELS[d.phase]||d.phase}}</span></div>`;
    if(d.maxComp) h+=`<div class="ttr"><span>Max complexity</span><span>${{d.maxComp}}</span></div>`;
    if(d.totalT) h+=`<div class="ttr"><span>Total iters T</span><span>${{d.totalT.toLocaleString()}}</span></div>`;
    if(d.actualTPerStream && Object.keys(d.actualTPerStream).length){{
      const parts=Object.entries(d.actualTPerStream)
        .map(([k,v])=>`${{k}}:${{Number(v).toLocaleString()}}`).join('  ');
      h+=`<div class="ttr"><span>Streams T</span><span style="font-size:9px">${{parts}}</span></div>`;
      if(d.actualTWinner!=null) h+=`<div class="ttr"><span>Winner T</span><span>${{Number(d.actualTWinner).toLocaleString()}}</span></div>`;
    }}
    if(d.K) h+=`<div class="ttr"><span>Pop K</span><span>${{d.K}}</span></div>`;
    if(d.runtime) h+=`<div class="ttr"><span>Runtime</span><span>${{d.runtime.toFixed(2)}}s</span></div>`;
    if(d.ops) h+=`<div class="ttr"><span>Ops</span><span style="font-size:9px">${{d.ops}}</span></div>`;
    if(d.finalExpr) h+=`<div class="ttr" style="flex-direction:column;gap:2px"><span>Winner formula</span><span style="font-family:var(--font-mono);font-size:8px;word-break:break-all">${{d.finalExpr}}</span></div>`;
    const altCount=[d.formulaOls,d.formulaNested,...Object.values(d.formulaPolyrat||{{}})].filter(Boolean).length;
    if(altCount>0) h+=`<div style="font-size:8px;color:var(--color-text-tertiary);margin-top:2px">+${{altCount}} alt formula${{altCount>1?'s':''}} (click to toggle)</div>`;
    if(d.notes) h+=`<div class="ttr"><span>Notes</span><span>${{d.notes}}</span></div>`;
  }} else {{
    h+=`<div style="color:var(--color-text-tertiary);font-size:9px;margin-top:4px">Not run yet — click to log</div>`;
  }}
  tt.innerHTML=h;tt.style.display='block';
  const rect=e.target.getBoundingClientRect();
  const container=document.querySelector('.root').getBoundingClientRect();
  tt.style.left=Math.min(rect.left-container.left+20,container.width-220)+'px';
  tt.style.top=(rect.bottom-container.top+4)+'px';
}}
function hideTT(){{document.getElementById('ttbox').style.display='none';}}

const tabsElement=document.getElementById('tabs');
if(tabsElement){{
  tabsElement.addEventListener('click',e=>{{
    const t=e.target.closest('.tab');if(!t)return;
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');activeCat=t.dataset.cat;buildTable();
  }});
}}

async function init(){{
  // Merge PRELOADED_DB (from Python run) with any manual additions stored in VS Code notebook
  let stored={{}};
  try{{
    const s=await window.storage.get('lb_v2',false);
    if(s&&s.value) stored=JSON.parse(s.value);
  }}catch(e){{}}
  DB=Object.assign({{}},PRELOADED_DB,stored);
  buildTable();
}}
init();"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DragonSR vs PySR — Leaderboard</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"
  onload="if(typeof buildTable==='function')buildTable();"></script>
<!-- Plotly (for interactive Search statistics & landscape charts) -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
{CSS_VARS}
{CSS}
</style>
</head>
<body>
{BODY}
<script>
{JS}
</script>
</body>
</html>"""

    out = Path(__file__).parent / "leaderboard_standalone.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ Standalone HTML written ({len(db)} entries): {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

