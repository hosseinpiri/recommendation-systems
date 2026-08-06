"""Generate numbers.tex (macros) from the three result JSONs, so the memo's
numbers are always exactly the pipeline's numbers."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


def f(x, d=3):
    out = f"{x:.{d}f}"
    return "0." + "0" * d if out == "-0." + "0" * d else out


def main():
    with open(os.path.join(OUT, "p1a_consumption.json")) as fh:
        a = json.load(fh)
    with open(os.path.join(OUT, "p1b_transition.json")) as fh:
        b = json.load(fh)
    sim = None
    p2 = os.path.join(OUT, "p2_simulation.json")
    if os.path.exists(p2):
        with open(p2) as fh:
            sim = json.load(fh)

    L = []
    def cmd(name, val):
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    cmd("nfit", f"{a['n_fit']:,}")
    cmd("ntest", f"{a['n_test']:,}")
    cmd("clickrate", f(100 * a['click_rate_fit'], 1))
    cmd("kappahat", f(a['kappa']))
    cmd("kappase", f(a['kappa_se']))
    cmd("kappacilo", f(a['kappa_ci95'][0]))
    cmd("kappacihi", f(a['kappa_ci95'][1]))
    cmd("LRkappa", f"{a['LR_kappa_vs_null']:,.0f}")
    cmd("aucholdout", f(a['auc_holdout']))

    cmd("nusers", f"{b['n_users']:,}")
    cmd("ndevrows", f"{b['n_dev_rows']:,}")
    cmd("ntrainclicks", f"{b['n_train_clicks']:,}")
    cmd("rhohat", f(b['rho_hat'], 3))
    lo, hi = b['rho_ci95_profileLR']
    if lo == hi:  # profile sharper than the grid: local quadratic approximation
        import numpy as np
        rg = np.array(b["rho_grid"])
        ll = np.array(b["profile_loglik"])
        near = np.abs(rg - b["rho_hat"]) <= 0.0101
        c2 = np.polyfit(rg[near], ll[near], 2)[0]
        delta = float(np.sqrt(1.92 / max(-c2, 1e-9)))
        lo, hi = b["rho_hat"] - delta, b["rho_hat"] + delta
    cmd("rhocilo", f(lo, 3))
    cmd("rhocihi", f(hi, 3))
    cmd("LRrho", f"{b['LR_rho_vs_zero']:,.1f}")
    cmd("kappadev", f(b['kappa_at_rho_hat_dev']))
    cmd("kappadevse", f(b['kappa_at_rho_hat_dev_se']))
    ns = b.get("ns_at_rhohat")
    if ns:  # rho_ns profile evaluated at the refined rho_hat (audit fix)
        cmd("rhonshat", f(ns['rho_ns_hat'], 3))
        cmd("LRns", f"{ns['LR_vs_zero']:,.1f}")
    else:
        cmd("rhonshat", f(b['rho_ns_hat'], 3))
        cmd("LRns", f"{b['LR_rho_ns_vs_zero']:,.1f}")
    cmd("runmeandelta", f(b['best_runmean_vs_best_const_rho'], 1))

    # alpha table body
    rows = sorted(a["alpha_by_category"].items(), key=lambda kv: -kv[1]["alpha"])
    body = "\n".join(
        f"{name} & {f(v['alpha'])} & {f(v['se'])} & {v['n_fit']:,} \\\\"
        for name, v in rows)
    with open(os.path.join(OUT, "alpha_table.tex"), "w") as fh:
        fh.write(body + "\n\\bottomrule\n")

    if sim:
        cfg = sim["config"]
        cmd("simH", cfg["H"])
        cmd("simEp", f"{cfg['N_EP']:,}")
        cmd("simItems", cfg["N_ITEMS"])
        rh = str(cfg["rho_hat"])
        res = sim["results"]

        def g(rho, pol, key="mean_clicks"):
            return res[str(rho)][pol][key]

        for tag, pol in [("oa", "oracle-aware"), ("ob", "oracle-blind"),
                         ("la", "learn-aware"), ("lb", "learn-blind")]:
            cmd(f"clk{tag}", f(g(rh, pol), 2))
            cmd(f"clkse{tag}", f(res[rh][pol]["se_over_seeds"], 3))
        gap_or = g(rh, "oracle-aware") - g(rh, "oracle-blind")
        gap_ln = g(rh, "learn-aware") - g(rh, "learn-blind")
        cmd("gapor", f(gap_or, 2))
        cmd("gaporpct", f(100 * gap_or / g(rh, "oracle-blind"), 1))
        cmd("gapln", f(gap_ln, 2))
        cmd("gaplnpct", f(100 * gap_ln / g(rh, "learn-blind"), 1))
        rhos = sorted(float(r) for r in res)
        rmax = str(rhos[-1])
        gapmax = g(rmax, "oracle-aware") - g(rmax, "oracle-blind")
        cmd("rhomax", f(rhos[-1], 2))
        cmd("gapmaxpct", f(100 * gapmax / g(rmax, "oracle-blind"), 1))
        cmd("bridgeshare", f(100 * res[rh]["oracle-aware"].get("bridge_share", 0), 1))
        cmd("bridgesharemax", f(100 * res[rmax]["oracle-aware"].get("bridge_share", 0), 1))
        val = sim["planner_validation"]
        worst = min(v["planner_mc"] / v["exact_dp"] for v in val.values())
        cmd("plannerworst", f(100 * worst, 1))
        # simulation results table body
        lines = []
        for rho in rhos:
            cells = [f(rho, 3)]
            for pol in ["oracle-aware", "oracle-blind", "learn-aware", "learn-blind"]:
                cells.append(f(g(rho, pol), 2))
            gp = 100 * (g(rho, "oracle-aware") - g(rho, "oracle-blind")) / g(rho, "oracle-blind")
            cells.append(f(gp, 1) + "\\%")
            lines.append(" & ".join(cells) + " \\\\")
        with open(os.path.join(OUT, "sim_table.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n\\bottomrule\n")

    # heterogeneous-margin scenario + horizon sweep
    ph = os.path.join(OUT, "p2_simulation_hetero.json")
    if os.path.exists(ph):
        with open(ph) as fh:
            het = json.load(fh)
        res = het["results"]
        rh = str(het["config"]["rho_hat"])
        rhos = sorted(float(r) for r in res)
        rmax = str(rhos[-1])

        def gr(rho, pol):
            return res[str(rho)][pol]["mean_revenue"]

        hgap = gr(rh, "oracle-aware") - gr(rh, "oracle-blind")
        hgapmax = gr(rmax, "oracle-aware") - gr(rmax, "oracle-blind")
        cmd("hgaporpct", f(100 * hgap / gr(rh, "oracle-blind"), 1))
        cmd("hgapmaxpct", f(100 * hgapmax / gr(rmax, "oracle-blind"), 1))
        cmd("hbridgeshare", f(100 * res[rh]["oracle-aware"].get("bridge_share", 0), 1))
        cmd("hbridgesharemax", f(100 * res[rmax]["oracle-aware"].get("bridge_share", 0), 1))
        lines = []
        for rho in rhos:
            cells = [f(rho, 3)]
            for pol in ["oracle-aware", "oracle-blind", "learn-aware", "learn-blind"]:
                cells.append(f(gr(rho, pol), 2))
            gp = 100 * (gr(rho, "oracle-aware") - gr(rho, "oracle-blind")) / gr(rho, "oracle-blind")
            cells.append(f(gp, 1) + "\\%")
            lines.append(" & ".join(cells) + " \\\\")
        with open(os.path.join(OUT, "sim_table_hetero.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n\\bottomrule\n")

    pw = os.path.join(OUT, "p2_hsweep.json")
    if os.path.exists(pw):
        with open(pw) as fh:
            hs = json.load(fh)
        r = hs["results"]["H=120"]["0.15"]
        g = 100 * (r["oracle-aware"]["mean_revenue"] - r["oracle-blind"]["mean_revenue"]) \
            / r["oracle-blind"]["mean_revenue"]
        cmd("hsweepgap", f(g, 1))

    pe = os.path.join(OUT, "p2_simulation_engagement.json")
    if os.path.exists(pe):
        with open(pe) as fh:
            eng = json.load(fh)
        res = eng["results"]
        gaps = []
        for rho in res:
            a = res[rho]["oracle-aware"]["mean_revenue"]
            bl = res[rho]["oracle-blind"]["mean_revenue"]
            gaps.append(100 * (a - bl) / bl)
        cmd("egapmaxpct", f(math.ceil(max(gaps) * 100) / 100, 2))

    pk = os.path.join(OUT, "p2_kappasweep.json")
    if os.path.exists(pk):
        with open(pk) as fh:
            ks = json.load(fh)
        rh = str(ks["config"]["rho_hat"])
        lines = []
        kgaps = {}
        for key in ks["results"]:
            kap = float(key.split("=")[1])
            row = ks["results"][key]
            gs = {}
            for rho in [rh, "0.15"]:
                a = row[rho]["oracle-aware"]["mean_revenue"]
                bl = row[rho]["oracle-blind"]["mean_revenue"]
                gs[rho] = 100 * (a - bl) / bl
            kgaps[kap] = gs
            lines.append(f"{f(kap, 2)} & {f(gs[rh], 1)}\\% & {f(gs['0.15'], 1)}\\% \\\\")
        with open(os.path.join(OUT, "kappa_table.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n\\bottomrule\n")
        kaps = sorted(kgaps)
        # threshold = smallest swept kappa whose rho=0.15 gap exceeds 2%
        thr = next((k for k in kaps if kgaps[k]["0.15"] >= 2.0), kaps[-1])
        cmd("kappathresh", f(thr, 0))
        cmd("kgapmidpct", f(kgaps[thr]["0.15"], 1))
        cmd("kgaphipct", f(kgaps[kaps[-1]]["0.15"], 1))

    # ---- follow-up batch: dense features, slate model, transitions, margins
    pemb = os.path.join(OUT, "p5_embeddings.json")
    if os.path.exists(pemb):
        with open(pemb) as fh:
            emb = json.load(fh)
        cmd("gramoffmean", f(emb["offdiag_mean"], 2))
        cmd("gramoffmin", f(emb["offdiag_min"], 2))
        cmd("gramoffmax", f(emb["offdiag_max"], 2))

    psl = os.path.join(OUT, "p5_slate_dense.json")
    if os.path.exists(psl):
        with open(psl) as fh:
            sl = json.load(fh)
        cmd("kapA", f(sl["A_cat"]["kappa"], 2))
        cmd("kapAsd", f(sl["A_cat"]["kappa_per_sd"], 2))
        cmd("kapB", f(sl["B_dense"]["kappa"], 2))
        cmd("kapBsd", f(sl["B_dense"]["kappa_per_sd"], 2))
        cmd("kapCsd", f(sl["C_dense_pos"]["kappa_per_sd"], 2))
        cmd("kapDcat", f(sl["D_slate_cat"]["kappa"], 2))
        cmd("kapDcatsd", f(sl["D_slate_cat"]["kappa_per_sd"], 2))
        cmd("kapDdensesd", f(sl["D_slate_dense"]["kappa_per_sd"], 2))
        cmd("posfar", f(abs(sl["C_dense_pos"]["pos_effects"]["bucket7"]), 2))
        cmd("singleclick", f(100 * sl.get("share_single_click_impressions",
                                          sl["share_single_click"]), 0))

    prr = os.path.join(OUT, "p5_rho_dense_renorm.json")
    if os.path.exists(prr):
        with open(prr) as fh:
            rr = json.load(fh)
        cmd("rhodensearticle", f(rr["article"]["rho_hat"], 2))
        cmd("LRdensearticle", f(rr["article"]["LR_vs_zero"], 0))

    p7 = os.path.join(OUT, "p7_sim_dense.json")
    if os.path.exists(p7):
        with open(p7) as fh:
            ds = json.load(fh)
        gaps = [abs(v["gap_pct"]) for v in ds["results"].values()]
        cmd("densegapmax", f(max(gaps), 2))
        b2 = max(v["bridge2_share_dense"] for v in ds["results"].values())
        cmd("bridgetwodense", f(100 * b2, 1))

    p8 = os.path.join(OUT, "p8_sim_transition.json")
    if os.path.exists(p8):
        with open(p8) as fh:
            tr = json.load(fh)
        r = tr["results"]

        def gk(tag, ten, rns):
            return r[f"{tag}|{ten}|rho_ns={rns}"]["gap_pct"]

        cmd("bfgapzero", f(gk("kappa_hat", "empirical_mix", 0.0), 2))
        cmd("bfgapmild", f(gk("kappa_hat", "empirical_mix", -0.002), 2))
        cmd("bfgapstrong", f(gk("kappa_hat", "empirical_mix", -0.015), 2))
        cmd("bfgapnew", f(gk("kappa_hat", "new_n0_5", -0.015), 2))
        cmd("bfgaptenured", f(gk("kappa_hat", "tenured_n0_40", -0.015), 2))
        cmd("bfgapkfive", f(gk("kappa_5", "empirical_mix", -0.015), 2))
        cmd("bfgapkfivezero", f(gk("kappa_5", "empirical_mix", 0.0), 2))
        val = tr["validation"]
        errs = [abs(v["aware_mc"] - v["exact_dp"]) / abs(v["exact_dp"])
                for v in val.values()]
        cmd("bfplannererr", f(100 * max(errs), 1))

    p9 = os.path.join(OUT, "p9_margin_sweep.json")
    if os.path.exists(p9):
        with open(p9) as fh:
            ms = json.load(fh)
        cmd("nmarginvec", ms["n_vectors"])
        cmd("margingapmax", f(ms["max_gap_pct"], 2))
        cmd("margingapmed", f(ms["median_gap_pct"], 2))
        cmd("margingapninety", f(ms["p90_gap_pct"], 2))
        best_rh = [row.get("gap_pct_at_rho_hat") for row in ms["rows"]
                   if "gap_pct_at_rho_hat" in row]
        if best_rh:
            cmd("margingapmaxrhohat", f(max(best_rh), 2))

    pa = os.path.join(OUT, "arc_master_results.json")
    if os.path.exists(pa):
        with open(pa) as fh:
            arc = json.load(fh)
        pr = arc["profiles"]
        cmd("kapnine", f(pr["u9|free"]["kappa"][pr["u9|free"]["rho"].index(
            pr["u9|free"]["rho_hat"])], 3))
        bt = arc["bootstrap"]
        cmd("bootlo", f(bt["rho_hat_pct"]["2.5"], 3))
        cmd("boothi", f(bt["rho_hat_pct"]["97.5"], 3))
        cmd("bootreps", bt["n_reps"])
        mains = [x for x in arc["sim"] if x["family"] == "main"]
        tb = max(x["gap_pct"] + 1.833 * x["gap_pct_se"] for x in mains)
        cmd("equivbound", f(tb, 2))
        gate = {(s["kappa"], s["recenter"], s["rho"]): s
                for s in arc["sim"] if s["family"] == "gate"}
        cmd("gaterawseven", f(gate[(7.0, "raw", 0.15)]["gap_pct"], 1))
        cmd("gateisoseven", f(gate[(7.0, "iso", 0.15)]["gap_pct"], 1))
        cmd("gaterawfive", f(gate[(5.0, "raw", 0.15)]["gap_pct"], 1))
        cmd("gateisofive", f(gate[(5.0, "iso", 0.15)]["gap_pct"], 1))
        back = {(s["kappa"], s["rho_ns"]): s
                for s in arc["sim"] if s["family"] == "back"}
        cmd("arcbffive", f(back[(5.0, -0.015)]["gap_pct"], 1))
        cmd("arcbffivezero", f(back[(5.0, 0.0)]["gap_pct"], 2))
        if "margins" in arc:
            mg = arc["margins"]
            cmd("arcnvec", f"{mg['n_vectors']:,}")
            cmd("arcmaxgap", f(mg["max_gap_pct"], 2))
            cmd("arcpninenine", f(mg["p99"], 2))
            cmd("arcmedgap", f(mg["median"], 2))

    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote numbers.tex with", len(L), "macros")


if __name__ == "__main__":
    main()
