"""
Lab 5: Adversarial Search - Minimax & Alpha-Beta Pruning
Tasks covered:
  1. Trace the supplied tree by hand using Minimax (printed trace table)
  2. Leaf-evaluation counter added to both algorithms
"""
import math

# ------------------------------------------------------------------
# Minimax with leaf counter
# ------------------------------------------------------------------
def minimax(depth, index, max_turn, scores, target, counter):
    if depth == target:
        counter[0] += 1          # one leaf evaluated
        return scores[index]
    if max_turn:
        return max(
            minimax(depth + 1, index * 2, False, scores, target, counter),
            minimax(depth + 1, index * 2 + 1, False, scores, target, counter),
        )
    return min(
        minimax(depth + 1, index * 2, True, scores, target, counter),
        minimax(depth + 1, index * 2 + 1, True, scores, target, counter),
    )


# ------------------------------------------------------------------
# Alpha-Beta with leaf counter
# ------------------------------------------------------------------
def alphabeta(depth, index, max_turn, scores, target, alpha, beta, counter):
    if depth == target:
        counter[0] += 1          # one leaf evaluated
        return scores[index]
    if max_turn:
        value = float("-inf")
        for i in range(2):
            value = max(value, alphabeta(depth + 1, index * 2 + i, False,
                                         scores, target, alpha, beta, counter))
            alpha = max(alpha, value)
            if alpha >= beta:
                break            # prune remaining child
        return value
    value = float("inf")
    for i in range(2):
        value = min(value, alphabeta(depth + 1, index * 2 + i, True,
                                     scores, target, alpha, beta, counter))
        beta = min(beta, value)
        if alpha >= beta:
            break                # prune remaining child
    return value


# ------------------------------------------------------------------
# Task 1: bottom-up trace table (what you do by hand)
# ------------------------------------------------------------------
def trace(scores, target):
    print("  Bottom-up Minimax trace:")
    level = list(scores)
    for d in range(target - 1, -1, -1):
        is_max = (d % 2 == 0)
        role = "MAX" if is_max else "MIN"
        nxt = []
        for k in range(0, len(level), 2):
            a, b = level[k], level[k + 1]
            v = max(a, b) if is_max else min(a, b)
            print(f"    depth {d} ({role}): {role.lower()}({a}, {b}) = {v}")
            nxt.append(v)
        level = nxt
    return level[0]


def run(name, scores):
    print("=" * 60)
    print(name)
    print("Scores:", scores)
    padded = list(scores)
    tree_depth = math.ceil(math.log2(len(padded)))
    padded += [0] * (2 ** tree_depth - len(padded))   # pad to complete tree
    if padded != list(scores):
        print("Padded:", padded)
    print("Tree depth:", tree_depth)

    traced = trace(padded, tree_depth)

    mm_count = [0]
    mm_val = minimax(0, 0, True, padded, tree_depth, mm_count)

    ab_count = [0]
    ab_val = alphabeta(0, 0, True, padded, tree_depth,
                       float("-inf"), float("inf"), ab_count)

    print(f"  Hand-trace root value : {traced}")
    print(f"  Minimax value         : {mm_val}   | leaves evaluated = {mm_count[0]}")
    print(f"  Alpha-Beta value      : {ab_val}   | leaves evaluated = {ab_count[0]}")
    assert mm_val == ab_val == traced, "Values must agree!"
    assert ab_count[0] <= mm_count[0], "Alpha-beta must not evaluate more leaves"
    print(f"  Leaves saved by pruning: {mm_count[0] - ab_count[0]}")


if __name__ == "__main__":
    # Tree from 5.1 (6 leaves, padded with zeros to 8)
    run("Tree A (from 5.1 Minimax code)", [3, 5, 2, 9, 0, 7])
    # Tree from 5.2
    run("Tree B (from 5.2 Alpha-Beta code)", [3, 5, 2, 9, 3, 5, 2, 9])
    # Edge case: all leaves equal -> no strict benefit, values still agree
    run("Edge case: all leaves equal", [4, 4, 4, 4, 4, 4, 4, 4])
