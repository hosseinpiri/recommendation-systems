"""
Multi-click ordering robustness (review item 9). The transition
z <- (1-rho) z + rho e_c is order-dependent within multi-click impressions;
MIND records a common impression timestamp, so within-impression order is a
data-processing convention. Re-run the rho profile under four conventions on
a 40k-user subsample: forward (baseline sort), reverse, random shuffle of
tied-timestamp clicks, and simultaneous (one update toward the average
category vector of the tied clicks).
"""
import json
import os
import numpy as np

import p1b_transition as T

RHOS = [0.0, 0.02, 0.035, 0.05, 0.08]
N_USERS = 40000


def reorder(events, mode, rng):
    """events: time-sorted [(t, c, isclick)]. Reorder clicks within equal t."""
    if mode == "forward":
        return events
    out = []
    i = 0
    while i < len(events):
        j = i
        while j < len(events) and events[j][0] == events[i][0]:
            j += 1
        block = list(events[i:j])
        clicks = [e for e in block if e[2] == 1]
        others = [e for e in block if e[2] != 1]
        if mode == "reverse":
            clicks = clicks[::-1]
        elif mode == "random":
            rng.shuffle(clicks)
        out.extend(others + clicks if mode != "forward" else block)
        i = j
    return out


def z_final_mode(z0, events, rho, mode, rng):
    z = z0.astype(np.float64).copy()
    if mode == "simultaneous":
        i = 0
        while i < len(events):
            j = i
            while j < len(events) and events[j][0] == events[i][0]:
                j += 1
            cats = [e[1] for e in events[i:j] if e[2] == 1]
            if cats and rho > 0:
                x = np.zeros(18)
                for c in cats:
                    x[c] += 1.0 / len(cats)
                z = (1 - rho) * z + rho * x
            i = j
        return z
    ev = reorder(events, mode, rng)
    for _, c, isclick in ev:
        if isclick and rho > 0:
            z *= (1 - rho)
            z[c] += rho
    return z


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    rng0 = np.random.default_rng(17)
    users = sorted(rng0.choice(users, N_USERS, replace=False).tolist())
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    cnt = np.bincount(all_cats, minlength=len(cat_list))
    keep_cats = np.where(cnt >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist, ev, dev, keep_cats, remap)

    # count multi-click tied blocks for context
    n_multi = 0
    n_clicks = 0
    for u in users:
        cs = [e for e in ev[u] if e[2] == 1]
        n_clicks += len(cs)
        times = {}
        for t, c, _ in cs:
            times[t] = times.get(t, 0) + 1
        n_multi += sum(v for v in times.values() if v > 1)
    share_tied = n_multi / max(n_clicks, 1)

    out = {"rhos": RHOS, "share_clicks_in_tied_blocks": share_tied,
           "profiles": {}}
    for mode in ("forward", "reverse", "random", "simultaneous"):
        rng = np.random.default_rng(99)
        lls = []
        for rho in RHOS:
            zmat = np.stack([z_final_mode(hist[u], ev[u], rho, mode, rng)
                             for u in D.users])
            ll, _ = D.maxll(zmat[D.uidx, D.cats])
            lls.append(ll)
            print(f"{mode} rho={rho}: {ll:.1f}", flush=True)
        i = int(np.argmax(lls))
        out["profiles"][mode] = {"loglik": lls, "rho_hat": RHOS[i],
                                 "LR_vs_zero": 2 * (lls[i] - lls[0])}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "p17_ordering.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("tied share:", round(share_tied, 3),
          {m: v["rho_hat"] for m, v in out["profiles"].items()})


if __name__ == "__main__":
    main()
