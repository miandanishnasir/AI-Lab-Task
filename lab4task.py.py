"""
Lab 4: Informed Search - Heuristics & A*
Rasikh Ali - 42365 | CAI601400 Artificial Intelligence Lab

Run:  python lab4_astar_audit.py
Output is printed and also saved to lab4_output.txt

Conventions (kept identical for every run, as the lab requires):
  * Priority queue entry = (f, g, counter, node, path)
      - ties on f are broken by smaller g, then by insertion order (counter)
  * Successors are examined in the order listed in the graph dict.
  * A node is "expanded" only when its successors are examined.
  * Goal processing is logged but NOT counted as an expansion.
  * Stale queue entries (g worse than best known g) are skipped, logged,
    and NOT counted.
"""
import heapq
import io
import sys

# --------------------------------------------------------------------------
# Problem definition (campus graph supplied in the lab manual)
# Edge weights are travel-time units; h values are abstract time estimates.
# --------------------------------------------------------------------------
GRAPH = {
    "S": [("A", 1), ("B", 4)],
    "A": [("C", 2)],
    "B": [("G", 5)],
    "C": [("G", 3)],
    "G": [],
}
H = {"S": 5, "A": 4, "B": 4, "C": 2, "G": 0}
START, GOAL = "S", "G"


# --------------------------------------------------------------------------
# Core search: one implementation serves both A* (h as given) and
# UCS (h = 0 for every node), so the comparison is a controlled one.
# --------------------------------------------------------------------------
def search(graph, h, start, goal):
    """Return dict with cost, path, expansions, expanded list and a trace."""
    counter = 0
    pq = [(h[start], 0, counter, start, [start])]
    best = {start: 0}
    expanded, trace = [], []
    stale = 0

    while pq:
        frontier_view = sorted(
            (f, g, n) for f, g, _, n, _ in pq
        )  # for logging only
        f, g, _, node, path = heapq.heappop(pq)

        if g != best[node]:
            stale += 1
            trace.append(f"  skip stale entry {node} (g={g}, best={best[node]})")
            continue

        if node == goal:
            trace.append(
                f"  pop {node}: g={g}, h={h[node]}, f={f}  -> GOAL reached "
                f"(logged, not counted as expansion)"
            )
            return {
                "cost": g, "path": path, "expansions": len(expanded),
                "expanded": expanded, "trace": trace, "stale": stale,
            }

        expanded.append(node)
        rest = [e for e in frontier_view if not (e[2] == node and e[1] == g)]
        trace.append(
            f"  expand {node}: g={g}, h={h[node]}, f={f}; "
            f"other frontier before expansion = "
            f"{[(n, 'g=' + str(gg), 'f=' + str(ff)) for ff, gg, n in rest]}"
        )
        for nxt, w in graph.get(node, []):
            ng = g + w
            if ng < best.get(nxt, float("inf")):
                best[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + h[nxt], ng, counter, nxt, path + [nxt]))
                trace.append(
                    f"      push {nxt}: g={g}+{w}={ng}, h={h[nxt]}, f={ng}+{h[nxt]}={ng + h[nxt]}"
                )
            else:
                trace.append(
                    f"      reject {nxt}: g={g}+{w}={ng} is not better than {best[nxt]}"
                )

    return {"cost": None, "path": None, "expansions": len(expanded),
            "expanded": expanded, "trace": trace, "stale": stale}


def path_cost(graph, path):
    """Independently sum the edge weights along a path."""
    total = 0
    for u, v in zip(path, path[1:]):
        total += dict(graph[u])[v]
    return total


# --------------------------------------------------------------------------
# True remaining costs (Dijkstra on the reversed graph) + heuristic audits
# --------------------------------------------------------------------------
def true_remaining(graph, goal):
    rev = {n: [] for n in graph}
    for u, edges in graph.items():
        for v, w in edges:
            rev[v].append((u, w))
    dist = {n: float("inf") for n in graph}
    dist[goal] = 0
    pq = [(0, goal)]
    while pq:
        d, n = heapq.heappop(pq)
        if d > dist[n]:
            continue
        for u, w in rev[n]:
            if d + w < dist[u]:
                dist[u] = d + w
                heapq.heappush(pq, (d + w, u))
    return dist


def audit(graph, h, goal, label):
    out = [f"Heuristic audit: {label}"]
    true_h = true_remaining(graph, goal)
    admissible = True
    out.append("  Admissibility  h(n) <= h*(n):")
    for n in graph:
        ok = h[n] <= true_h[n]
        admissible &= ok
        out.append(f"    {n}: h={h[n]}  h*={true_h[n]}  {'OK' if ok else 'OVERESTIMATE'}")
    consistent = True
    out.append("  Consistency  h(u) <= cost(u,v) + h(v):")
    for u, edges in graph.items():
        for v, w in edges:
            ok = h[u] <= w + h[v]
            consistent &= ok
            out.append(f"    {u}->{v}: {h[u]} <= {w} + {h[v]} = {w + h[v]}  {'OK' if ok else 'VIOLATED'}")
    out.append(f"  => admissible={admissible}, consistent={consistent}")
    return "\n".join(out), admissible, consistent


def exhaustive_routes(graph, start, goal):
    """All simple routes with costs (used to confirm the optimum)."""
    routes = []

    def dfs(n, p, c):
        if n == goal:
            routes.append((c, p))
            return
        for v, w in graph.get(n, []):
            if v not in p:
                dfs(v, p + [v], c + w)

    dfs(start, [start], 0)
    return sorted(routes)


def report_run(title, graph, h, start, goal):
    res = search(graph, h, start, goal)
    lines = [f"--- {title} ---", f"h = {h}"]
    lines += res["trace"]
    if res["path"] is None:
        lines.append(f"  result: no path found; expansions={res['expansions']}")
    else:
        lines.append(
            f"  result: path={'->'.join(res['path'])}, returned cost={res['cost']}, "
            f"independent sum={path_cost(graph, res['path'])}"
        )
        lines.append(
            f"  expanded nodes (in order) = {res['expanded']}, "
            f"expansion count = {res['expansions']}, stale entries skipped = {res['stale']}"
        )
    return "\n".join(lines), res


def main():
    buf = io.StringIO()

    def say(s=""):
        print(s)
        buf.write(s + "\n")

    say("=" * 72)
    say("LAB 4 - A* HEURISTIC AUDIT")
    say("=" * 72)

    # Ground truth
    say("\n[0] Ground truth: all routes S -> G")
    for c, p in exhaustive_routes(GRAPH, START, GOAL):
        say(f"  {'->'.join(p)} cost {c}")

    # Task 1: audit + manual frontier values
    say("\n[1] Heuristic audit (supplied h)")
    txt, adm, con = audit(GRAPH, H, GOAL, "supplied h")
    say(txt)

    # Task 2: A* vs UCS
    say("\n[2] A* vs UCS (same graph, same endpoints, same tie-breaking)")
    zero = {n: 0 for n in GRAPH}
    t_a, r_astar = report_run("A* with supplied h", GRAPH, H, START, GOAL)
    t_u, r_ucs = report_run("UCS control (h = 0)", GRAPH, zero, START, GOAL)
    say(t_a)
    say()
    say(t_u)
    say("\nComparison table")
    say(f"  {'Algorithm':<12}{'Path':<12}{'Cost':<6}{'Expanded (in order)':<22}{'Count':<6}")
    for name, r in (("A*", r_astar), ("UCS", r_ucs)):
        say(f"  {name:<12}{'->'.join(r['path']):<12}{r['cost']:<6}{','.join(r['expanded']):<22}{r['expansions']:<6}")

    # Task 3: overestimate
    say("\n[3] Overestimate experiment: change only h(C) from 2 to 10")
    true_h = true_remaining(GRAPH, GOAL)
    say(f"  true remaining cost of C = {true_h['C']}; new h(C) = 10 -> overestimates by {10 - true_h['C']}")
    h_bad = dict(H)
    h_bad["C"] = 10
    txt, adm_b, con_b = audit(GRAPH, h_bad, GOAL, "h(C)=10")
    say(txt)
    t_b, r_bad = report_run("A* with h(C)=10", GRAPH, h_bad, START, GOAL)
    say(t_b)
    say(
        f"  Changed? path {'->'.join(r_astar['path'])} -> {'->'.join(r_bad['path'])}, "
        f"cost {r_astar['cost']} -> {r_bad['cost']}, expansions {r_astar['expansions']} -> {r_bad['expansions']}"
    )

    # Extra: an overestimate that does NOT change the answer
    say("\n[3b] Counter-check: overestimate h(B)=10 (true 5) - result may look unchanged")
    h_b = dict(H)
    h_b["B"] = 10
    t_b2, r_b2 = report_run("A* with h(B)=10", GRAPH, h_b, START, GOAL)
    say(t_b2)
    say("  Optimal cost is still 6 here, so one unchanged run does NOT prove a heuristic safe.")

    # Edge cases
    say("\n[4] Edge cases")
    t_e1, r_e1 = report_run("Edge 1: start == goal", GRAPH, H, "G", "G")
    say(t_e1)
    g2 = {"S": [("A", 1)], "A": [], "G": []}
    h2 = {"S": 1, "A": 0, "G": 0}
    t_e2, r_e2 = report_run("Edge 2: goal unreachable", g2, h2, "S", "G")
    say(t_e2)
    # Stale entry case: a cheaper route to a node discovered later
    g3 = {"S": [("A", 5), ("B", 1)], "B": [("A", 1)], "A": [("G", 10)], "G": []}
    h3 = {"S": 0, "A": 0, "B": 0, "G": 0}
    t_e3, r_e3 = report_run("Edge 3: better path to A found later (stale entry)", g3, h3, "S", "G")
    say(t_e3)

    # Self-checks
    assert r_astar["cost"] == 6 and r_astar["path"] == ["S", "A", "C", "G"]
    assert r_astar["expanded"] == ["S", "A", "C"] and r_astar["expansions"] == 3
    assert r_ucs["cost"] == 6 and r_ucs["expanded"] == ["S", "A", "C", "B"]
    assert r_bad["cost"] == 9 and r_bad["path"] == ["S", "B", "G"]
    assert r_b2["cost"] == 6
    assert r_e1["cost"] == 0 and r_e2["path"] is None and r_e3["cost"] == 12 and r_e3["stale"] == 1
    say("\nAll self-checks passed.")

    with open("lab4_output.txt", "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())


if __name__ == "__main__":
    main()
