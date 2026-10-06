#!/usr/bin/env python
"""Amendment 3, package A3-4 (triggers and review): simulation runs.

Reads code/configs/osrev_triggers.yaml and writes raw, additive outputs to
results/2026-10-05_osrev/triggers/raw/. Run through code/slurm/osrev_triggers.sh
(one node, chunk-level parallel workers); `--n-patients` and `--workers`
exist for tests and pilots only. Tables are derived by
code/21_osrev_triggers_tables.py on the login node.

Blocks
  A  (parts a, b, c) four views, E4 base cell: the published cohort (library seed)
     and 10 new cohorts. Per chunk: histogram of per-patient (TP, FP, FN) pair
     counts for every error definition, trigger threshold and axis; patient-level
     2 by 2 counts; a mutually exclusive classification of pairs; between-axis
     differences.
  D  (part d) three views: reviewed rule over the reviewer-probability grid on
     common decision uniforms, 4 view-accuracy scenarios x 3 overestimation
     levels, plus the published cells rerun with their library seeds.
  E  (part e) four views: alternative joint AP/SL distributions on common random
     numbers (quantile-mapped ratio, shared measurement stream).
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lib"))
from duomaxsim import config as C  # noqa: E402
from duomaxsim import osrev_triggers as T  # noqa: E402
from duomaxsim.experiments import read_once  # noqa: E402
from duomaxsim.model import AXES, draw_latent  # noqa: E402
from duomaxsim.rules import a4_final  # noqa: E402

SPEC = yaml.safe_load(open(ROOT / "code" / "configs" / "osrev_triggers.yaml"))
META = SPEC["meta"]
ENT = int(META["seed_entropy"])
PRE = int(META["spawn_prefix"])
CHUNK = int(META["chunk_size"])
THR = float(META["error_threshold_mm"])
TAUS = [float(t) for t in META["taus_mm"]]
CFG = {"base": C.load_config(ROOT / META["base_config"]), "amend2": C.load_config(ROOT / META["amend2_config"])}
SCOPES = ("AP", "SL", "either")


def lib_cell(config: str, experiment: str, cell: int):
    """Parameters and SeedSequence of a library grid cell."""
    cfg = CFG[config]
    return C.cell_params(cfg, experiment, int(cell)), C.cell_seed(cfg, experiment, int(cell))


def seed_desc(ss: np.random.SeedSequence) -> str:
    return f"SeedSequence(entropy={ss.entropy}, spawn_key={tuple(int(k) for k in ss.spawn_key)})"


# =========================================================================== block A

def a_conditions():
    t = SPEC["triggers"]
    pub = t["published"]
    p, ss = lib_cell(pub["config"], pub["experiment"], pub["cell"])
    conds = [dict(block="A", cohort="published", rep=-1, p=p, entropy=int(ss.entropy), key=tuple(ss.spawn_key))]
    for rep in range(int(t["replicates"])):
        conds.append(dict(block="A", cohort="new", rep=rep, p=p, entropy=ENT, key=(PRE, 1, rep)))
    return conds


def _scope_any(x, scope):
    """(n, 2, m) pair booleans -> (n,) 'any pair' within the scope."""
    if scope == "either":
        return x.reshape(x.shape[0], -1).any(1)
    return x[:, AXES.index(scope)].any(-1)


def _scope_flag(x, scope):
    """(n, 2) per-axis booleans -> (n,)."""
    return x.any(1) if scope == "either" else x[:, AXES.index(scope)]


def work_A(cond, i, n_total):
    p = cond["p"]
    ss = np.random.SeedSequence(cond["entropy"], spawn_key=cond["key"])
    _, rc = next(T.cohort_chunks(p, ss, n_total, CHUNK, only=i))
    lat, vm, U = rc["lat"], rc["vm"], rc["U"]
    n = lat.n
    a4 = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0]
    est = {"A1_anchor": vm[..., 0], "A3_largest": vm.max(-1), "A4_reviewed": a4}
    wrong = {k: np.abs(v - lat.S) > THR for k, v in est.items()}
    err = T.view_error_mm(lat)
    pat, pl, part, exam = {}, {}, {}, {}
    for tau in TAUS:
        fired = T.trigger_fired(vm, tau)
        defs = T.pair_event_definitions(lat, THR, tau)
        for name, ev in defs.items():
            for d in range(2):
                pat[(tau, name, AXES[d])] = T.pattern_hist(fired[:, d], ev[:, d])
            for sc in SCOPES:
                f_any, e_any = _scope_any(fired, sc), _scope_any(ev, sc)
                c = T.two_by_two(f_any, e_any)
                c["tp_matched"] = int(_scope_any(fired & ev, sc).sum())   # fired in a pair that carries the error
                pl[(tau, "any_pair:" + name, sc)] = c
        for k, w in wrong.items():
            for sc in SCOPES:
                c = T.two_by_two(_scope_any(fired, sc), _scope_flag(w, sc))
                c["tp_matched"] = -1
                pl[(tau, "reported_wrong:" + k, sc)] = c
        for kind in ("total", "net"):
            code = T.pair_partition(lat, THR, kind)
            dexp = err["net"][..., 1:] - err["net"][..., :1]
            for d in range(2):
                for cl in range(len(T.PARTITION_LABELS)):
                    m_ = code[:, d] == cl
                    part[(tau, kind, AXES[d], cl)] = dict(
                        n_pairs=int(m_.sum()), n_fired=int((fired[:, d] & m_).sum()),
                        n_agree_within_tau=int((m_ & (np.abs(dexp[:, d]) < tau)).sum()),
                        n_patients_any=int(m_.any(1).sum()),
                        n_patients_any_fired_any=int((m_.any(1) & fired[:, d].any(1)).sum()))
            # whole-examination offsets: every view of the axis wrong in the same direction
            e = err[kind]
            for d in range(2):
                low = (e[:, d] < -THR).all(1)
                high = (e[:, d] > THR).all(1)
                fa = fired[:, d].any(1)
                exam[(tau, kind, AXES[d])] = dict(
                    n=n, all_low=int(low.sum()), all_low_fired=int((low & fa).sum()),
                    all_high=int(high.sum()), all_high_fired=int((high & fa).sum()),
                    any_view_wrong=int((np.abs(e[:, d]) > THR).any(1).sum()),
                    any_view_wrong_fired=int(((np.abs(e[:, d]) > THR).any(1) & fa).sum()),
                    fired_any=int(fa.sum()))
    # between-axis differences (the published quantities use the anchor view)
    ax = {}
    true_d = np.abs(lat.S[:, 0] - lat.S[:, 1])
    for name, v in est.items():
        md = np.abs(v[:, 0] - v[:, 1])
        for tau in TAUS:
            ax[("measured:" + name, tau)] = int((md >= tau).sum())
    for tau in TAUS:
        ax[("true", tau)] = int((true_d >= tau).sum())
    return dict(n=n, pat=pat, pl=pl, part=part, exam=exam, ax=ax,
                instr=dict(n=n, n_g_shift_gt_thr_ap=int((np.abs((lat.g - 1) * lat.S[:, 0]) > THR).sum()),
                           n_g_shift_gt_thr_sl=int((np.abs((lat.g - 1) * lat.S[:, 1]) > THR).sum())))


def merge_A(cond, parts):
    """Sum chunk outputs of one cohort into tidy rows."""
    key = dict(cohort=cond["cohort"], rep=cond["rep"], seed=f"({cond['entropy']}, {tuple(cond['key'])})")
    n = sum(q["n"] for q in parts)
    rows = defaultdict(list)
    pat = defaultdict(lambda: 0)
    for q in parts:
        for k, H in q["pat"].items():
            pat[k] = pat[k] + H
    for (tau, name, axis), H in pat.items():
        m = H.shape[0] - 1
        for a, b, c in itertools.product(range(m + 1), repeat=3):
            if H[a, b, c]:
                rows["A_pair_patterns"].append(dict(**key, axis=axis, tau_mm=tau, definition=name, tp_pairs=a,
                                                    fp_pairs=b, fn_pairs=c, tn_pairs=m - a - b - c,
                                                    n_patients=int(H[a, b, c])))

    def summed(field):
        acc = defaultdict(lambda: defaultdict(int))
        for q in parts:
            for k, dct in q[field].items():
                for kk, v in dct.items():
                    acc[k][kk] += v
        return acc

    for (tau, truth, scope), c in summed("pl").items():
        if truth.startswith("reported_wrong"):
            c["tp_matched"] = -1
        rows["A_patient_2x2"].append(dict(**key, scope=scope, tau_mm=tau, truth=truth, **c, n_patients=n))
    for (tau, kind, axis, cl), c in summed("part").items():
        rows["A_pair_partition"].append(dict(**key, axis=axis, tau_mm=tau, error_kind=kind, class_id=cl,
                                             class_label=T.PARTITION_LABELS[cl], **c, n_patients=n))
    for (tau, kind, axis), c in summed("exam").items():
        c.pop("n")
        rows["A_exam_offsets"].append(dict(**key, axis=axis, tau_mm=tau, error_kind=kind, **c, n_patients=n))
    ax = defaultdict(int)
    for q in parts:
        for k, v in q["ax"].items():
            ax[k] += v
    for (quantity, tau), k in ax.items():
        rows["A_axis_diff"].append(dict(**key, quantity=quantity, tau_mm=tau, k=int(k), n_patients=n))
    ins = defaultdict(int)
    for q in parts:
        for k, v in q["instr"].items():
            ins[k] += v
    rows["A_instrument"].append(dict(**key, **{k: int(v) for k, v in ins.items() if k != "n"}, n_patients=n))
    return rows


# =========================================================================== block D

def review_grid():
    r = SPEC["review"]
    g = [(float(s), float(f)) for s in r["s_det"] for f in r["f_rej"]]
    for q in r["uninformative"]:
        if (float(q), float(q)) not in g:
            g.append((float(q), float(q)))
    return g


def d_conditions():
    r = SPEC["review"]
    conds = []
    for (sn, sov), (on, oov) in itertools.product(r["scenarios"].items(), r["overestimation"].items()):
        p = C.base_params(CFG["base"])
        p.update(r["fixed"])
        p.update(sov)
        p.update(oov)
        conds.append(dict(block="D", run="new", scenario=sn, overestimation=on, p=p, entropy=ENT,
                          key=tuple(r["seed_key"]), source="base.yaml + overrides"))
    for rp in r["reproduce"]:
        p, ss = lib_cell(rp["config"], rp["experiment"], rp["cell"])
        conds.append(dict(block="D", run="reproduce", scenario=rp["label"], overestimation="base", p=p,
                          entropy=int(ss.entropy), key=tuple(ss.spawn_key),
                          source=f"{rp['config']} {rp['experiment']} cell {rp['cell']}"))
    return conds


def work_D(cond, i, n_total):
    p = cond["p"]
    ss = np.random.SeedSequence(cond["entropy"], spawn_key=cond["key"])
    _, rc = next(T.cohort_chunks(p, ss, n_total, CHUNK, only=i))
    lat, vm, U = rc["lat"], rc["vm"], rc["U"]
    grid = review_grid()
    res = T.reviewed_rule_grid(vm, lat.over, U, p["t_warn"], p["t_adj"], grid, p["a4_scrutiny"])
    a3 = vm.max(-1)
    if not np.array_equal(res[(0.0, 0.0)]["value"], a3):
        raise AssertionError("reviewed rule with (0, 0) differs from the largest view mean")
    names = ["A1", "A2", "A3"] + [f"A4|{s:g}|{f:g}" for s, f in grid]
    est = np.stack([vm[..., 0], vm.mean(-1), a3] + [res[g]["value"] for g in grid], axis=1)   # (n, V, 2)
    proc = {}
    for g in grid:
        r = res[g]
        for d in range(2):
            over_o = lat.over[:, d, 1:]
            proc[(g, AXES[d])] = dict(
                n_other_views=int(r["reviewed"][:, d].size),
                n_reviewed=int(r["reviewed"][:, d].sum()),
                n_reviewed_over=int((r["reviewed"][:, d] & over_o).sum()),
                n_excluded=int(r["reject"][:, d].sum()),
                n_excluded_over=int(r["reject_over"][:, d].sum()),
                n_excluded_valid=int(r["reject_valid"][:, d].sum()),
                n_patients_reviewed=int(r["reviewed"][:, d].any(1).sum()),
                n_patients_excluded=int(r["reject"][:, d].any(1).sum()),
                n_patients_value_changed=int((r["value"][:, d] != a3[:, d]).sum()),
                n_over_other_views=int(over_o.sum()))
    return dict(n=lat.n, names=names, est=est, S=lat.S, proc=proc)


def merge_D(cond, parts):
    r = SPEC["review"]
    cuts = [float(c) for c in r["cutoffs_mm"]]
    names = parts[0]["names"]
    est = np.concatenate([q["est"] for q in parts])
    S = np.concatenate([q["S"] for q in parts])
    key = dict(run=cond["run"], scenario=cond["scenario"], overestimation=cond["overestimation"],
               source=cond["source"], seed=f"({cond['entropy']}, {tuple(cond['key'])})",
               u_scale=str(list(cond["p"]["u_scale"])), window=cond["p"]["window"], K=int(cond["p"]["K"]),
               view_over=bool(cond["p"]["view_over"]), p_over=str(list(cond["p"]["p_over"])))
    rows = defaultdict(list)

    def split(nm):
        if nm.startswith("A4|"):
            _, s, f = nm.split("|")
            return "A4", float(s), float(f)
        return nm, np.nan, np.nan

    i3 = names.index("A3")
    ib = names.index("A4|0.8|0.1")
    for d in range(2):
        for j, nm in enumerate(names):
            rule, s, f = split(nm)
            k2 = dict(**key, axis=AXES[d], rule=rule, s_det=s, f_rej=f)
            for metric, (v, se, nn, kk) in T.error_metrics(est[:, j, d], S[:, d], cuts).items():
                rows["D_review_metrics"].append(dict(**k2, metric=metric, value=v, mcse=se, n=nn, k=kk))
            for ref_name, ir in (("A3", i3), ("A4|0.8|0.1", ib)):
                if rule != "A4" and not (ref_name == "A3" and rule == "A1"):
                    continue
                for metric, (v, se, nn) in T.paired_contrast(est[:, j, d], est[:, ir, d], S[:, d], cuts).items():
                    rows["D_review_paired"].append(dict(**k2, minus=ref_name, metric=metric, difference=v,
                                                        mcse=se, n=nn))
    proc = defaultdict(lambda: defaultdict(int))
    for q in parts:
        for k, dct in q["proc"].items():
            for kk, v in dct.items():
                proc[k][kk] += v
    for ((s, f), axis), c in proc.items():
        rows["D_review_process"].append(dict(**key, axis=axis, s_det=s, f_rej=f, **c, n_patients=len(S)))
    return rows


# =========================================================================== block E

def e_scenarios(p):
    a = SPEC["axes"]
    sc = {}
    for diff in a["mean_difference_mm"]:
        for lab, c in a["beta_concentration"].items():
            sc[f"beta_diff{diff:g}_{lab}"] = dict(T.spec_for_mean_difference(p, "beta", diff, r_concentration=c),
                                                  target_mean_diff_mm=diff, variability=lab)
        for lab, rho in a["lognormal_log_corr"].items():
            sc[f"lognormal_diff{diff:g}_{lab}"] = dict(T.spec_for_mean_difference(p, "lognormal", diff, log_corr=rho),
                                                       target_mean_diff_mm=diff, variability=lab, log_corr_target=rho)
    for lab, spec in a["extra"].items():
        sc[lab] = dict(spec)
    return sc


def e_conditions():
    pub = SPEC["triggers"]["published"]
    p, _ = lib_cell(pub["config"], pub["experiment"], pub["cell"])
    return [dict(block="E", scenario=name, spec=spec, p=p) for name, spec in e_scenarios(p).items()]


E_SPEC_KEYS = ("family", "r_mean", "r_lower", "r_upper", "r_concentration", "mu_log", "sd_log", "value")


def work_E(cond, i, n_total):
    p = cond["p"]
    K = int(p["K"])
    spec = {k: v for k, v in cond["spec"].items() if k in E_SPEC_KEYS}
    nch = T.n_chunks(n_total, CHUNK)
    n = min(CHUNK, n_total - i * CHUNK)
    lat0 = draw_latent(p, n, T.gen(np.random.SeedSequence(ENT, spawn_key=(PRE, 5, 0)).spawn(nch)[i]))
    lat = T.rescale_sl(lat0, T.ratio_map(lat0.r, p, spec))
    rng = T.gen(np.random.SeedSequence(ENT, spawn_key=(PRE, 5, 1)).spawn(nch)[i])
    rb = p["sigma_rb_mm"] * rng.standard_normal(n)
    _, br = read_once(lat, p, rng, rb)
    vm = br.value
    U = rng.random((n, 2, K - 1))
    a4 = a4_final(vm, lat.over, U, p["t_warn"], p["t_adj"], p["s_det"], p["f_rej"], p["a4_scrutiny"])[0]
    est = {"A1_anchor": vm[..., 0], "A3_largest": vm.max(-1), "A4_reviewed": a4}
    sd = lat.S[:, 0] - lat.S[:, 1]
    la, ls = np.log(lat.S[:, 0]), np.log(lat.S[:, 1])
    out = dict(n=n, cnt=defaultdict(int), sums={})
    for tau in TAUS:
        out["cnt"][("true", tau)] = int((np.abs(sd) >= tau).sum())
        for k, v in est.items():
            out["cnt"][("measured:" + k, tau)] = int((np.abs(v[:, 0] - v[:, 1]) >= tau).sum())
        fired = T.trigger_fired(vm, tau)
        ev = T.pair_event_definitions(lat, THR)["D0_published"]
        for d in range(2):
            out["cnt"][("any_fired:" + AXES[d], tau)] = int(fired[:, d].any(1).sum())
            for kk, vv in T.two_by_two(fired[:, d], ev[:, d]).items():
                out["cnt"][(f"pair_{kk}:" + AXES[d], tau)] = vv
    r = lat.r
    amd = np.abs(est["A1_anchor"][:, 0] - est["A1_anchor"][:, 1])
    out["sums"] = dict(d=sd.sum(), d2=(sd * sd).sum(), ad=np.abs(sd).sum(), ad2=(sd * sd).sum(),
                       amd=amd.sum(), amd2=(amd * amd).sum(),
                       r=r.sum(), r2=(r * r).sum(), sl_gt_ap=float((sd < 0).sum()),
                       la=la.sum(), ls=ls.sum(), laa=(la * la).sum(), lss=(ls * ls).sum(), las=(la * ls).sum(),
                       a=lat.S[:, 0].sum(), s=lat.S[:, 1].sum(), aa=(lat.S[:, 0] ** 2).sum(),
                       ss=(lat.S[:, 1] ** 2).sum(), as_=(lat.S[:, 0] * lat.S[:, 1]).sum())
    # AP-axis invariance: these checksums must be identical in every scenario
    out["ap_check"] = (float(vm[:, 0].sum()), float(lat.spans[:, 0].sum()), float(lat.S[:, 0].sum()))
    return out


def merge_E(cond, parts):
    n = sum(q["n"] for q in parts)
    spec = cond["spec"]
    key = dict(scenario=cond["scenario"], **{k: spec.get(k, np.nan) for k in E_SPEC_KEYS},
               target_mean_diff_mm=spec.get("target_mean_diff_mm", np.nan),
               variability=spec.get("variability", ""), log_corr_target=spec.get("log_corr_target", np.nan),
               seed_latent=f"({ENT}, ({PRE}, 5, 0))", seed_measure=f"({ENT}, ({PRE}, 5, 1))")
    rows = defaultdict(list)
    cnt = defaultdict(int)
    for q in parts:
        for k, v in q["cnt"].items():
            cnt[k] += v
    pspec = {k: v for k, v in spec.items() if k in E_SPEC_KEYS}
    for (quantity, tau), k in cnt.items():
        ana = T.axis_prob_ge(cond["p"], pspec, tau) if quantity == "true" else np.nan
        rows["E_axis_counts"].append(dict(**key, quantity=quantity, tau_mm=tau, k=int(k), n_patients=n,
                                          analytic=ana))
    s = defaultdict(float)
    for q in parts:
        for k, v in q["sums"].items():
            s[k] += float(v)

    def corr(sx, sy, sxx, syy, sxy):
        vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
        if vx <= 0 or vy <= 0:
            return np.nan
        return (sxy / n - sx * sy / n ** 2) / np.sqrt(vx * vy)

    def msd(s1, s2):
        m = s1 / n
        return m, np.sqrt(max(s2 / n - m * m, 0.0) * n / (n - 1))

    md, sdd = msd(s["d"], s["d2"])
    mad, sdad = msd(s["ad"], s["ad2"])
    mam, sdam = msd(s["amd"], s["amd2"])
    mr, sdr = msd(s["r"], s["r2"])
    rows["E_axis_summary"].append(dict(
        **key, n_patients=n, mean_true_diff_mm=md, mean_true_diff_mcse=sdd / np.sqrt(n), sd_true_diff_mm=sdd,
        mean_abs_true_diff_mm=mad, mean_abs_true_diff_mcse=sdad / np.sqrt(n),
        mean_abs_anchor_measured_diff_mm=mam, mean_abs_anchor_measured_diff_mcse=sdam / np.sqrt(n),
        mean_ratio=mr, sd_ratio=sdr, n_sl_gt_ap=int(s["sl_gt_ap"]),
        corr_log=corr(s["la"], s["ls"], s["laa"], s["lss"], s["las"]),
        corr_mm=corr(s["a"], s["s"], s["aa"], s["ss"], s["as_"]),
        ap_check_vm=repr(sum(q["ap_check"][0] for q in parts)),
        ap_check_spans=repr(sum(q["ap_check"][1] for q in parts)),
        ap_check_S=repr(sum(q["ap_check"][2] for q in parts))))
    return rows


# =========================================================================== driver

WORK = {"A": work_A, "D": work_D, "E": work_E}
MERGE = {"A": merge_A, "D": merge_D, "E": merge_E}


def _run(task):
    ci, cond, i, n_total = task
    t0 = time.process_time()
    out = WORK[cond["block"]](cond, i, n_total)
    return ci, i, out, time.process_time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "2026-10-05_osrev" / "triggers" / "raw"))
    ap.add_argument("--n-patients", type=int, default=int(META["n_patients"]))
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--blocks", default="ADE")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    conds = [c for c in a_conditions() + d_conditions() + e_conditions() if c["block"] in a.blocks]
    nch = T.n_chunks(a.n_patients, CHUNK)
    tasks = [(ci, c, i, a.n_patients) for ci, c in enumerate(conds) for i in range(nch)]
    print(f"conditions {len(conds)}, chunk tasks {len(tasks)}, workers {a.workers}, n per condition {a.n_patients}",
          flush=True)
    w0 = time.perf_counter()
    got = defaultdict(dict)
    cpu = 0.0
    tables = defaultdict(list)
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        for ci, i, res, c in pool.imap_unordered(_run, tasks, chunksize=1):
            got[ci][i] = res
            cpu += c
            if len(got[ci]) == nch:
                cond = conds[ci]
                parts = [got[ci][j] for j in range(nch)]     # chunk order, as in the library
                for name, rows in MERGE[cond["block"]](cond, parts).items():
                    tables[name] += rows
                del got[ci]
    if "E" in a.blocks:
        chk = {(r["ap_check_vm"], r["ap_check_spans"], r["ap_check_S"]) for r in tables["E_axis_summary"]}
        if len(chk) != 1:
            raise AssertionError("AP-axis outputs differ between AP/SL scenarios")
        print("AP-axis checksums identical across", len(tables["E_axis_summary"]), "AP/SL scenarios")
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        sort = [c for c in ("run", "cohort", "rep", "scenario", "overestimation", "axis", "scope", "tau_mm",
                            "definition", "truth", "error_kind", "class_id", "quantity", "rule", "s_det", "f_rej",
                            "minus", "metric", "tp_pairs", "fp_pairs", "fn_pairs") if c in df.columns]
        df.sort_values(sort, kind="stable").to_csv(out / f"{name}.csv", index=False)
        print("wrote", name, df.shape, flush=True)
    print(f"cpu_s {cpu:.1f} wall_s {time.perf_counter() - w0:.1f}")


if __name__ == "__main__":
    main()
