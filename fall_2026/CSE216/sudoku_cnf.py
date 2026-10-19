
from itertools import combinations

PUZZLE = [
    "800000000",
    "003600000",
    "070090200",
    "050007000",
    "000045700",
    "000100030",
    "001000068",
    "008500010",
    "090000400",
]
"""
solved

812 753 649
943 682 175
675 491 283

154 237 896
369 845 721
287 169 534

521 974 368
438 526 917
796 318 452

"""


def var(r, c, v):
    return 81 * r + 9 * c + v + 1


def main():
    i=input("decode")
    if i=="decode":
        grid = [[0] * 9 for _ in range(9)]
        for line in open("sol.txt"):
            if line.startswith("v "):
                for n in map(int, line.split()[1:]):
                    if 0 < n <= 729:
                        n -= 1
                        grid[n // 81][n // 9 % 9] = n % 9 + 1
        for row in grid:
            print("".join(map(str, row)))
        return

    clauses = []

    # each cell holds exactly one value
    for r in range(9):
        for c in range(9):
            clauses.append([var(r, c, v) for v in range(9)])
            for a, b in combinations(range(9), 2):
                clauses.append([-var(r, c, a), -var(r, c, b)])

    #each value appears exactly once in every row, column and box
    rows = [[(r, c) for c in range(9)] for r in range(9)]
    cols = [[(r, c) for r in range(9)] for c in range(9)]
    #3x3s
    boxes = [[(3 * br + i, 3 * bc + j) for i in range(3) for j in range(3)]
            for br in range(3) for bc in range(3)]

    for group in rows + cols + boxes:
        for v in range(9):
            clauses.append([var(r, c, v) for r, c in group])
            for (r1, c1), (r2, c2) in combinations(group, 2):
                clauses.append([-var(r1, c1, v), -var(r2, c2, v)])

    # givens
    for r, row in enumerate(PUZZLE):
        for c, ch in enumerate(row):
            if ch != "0":
                clauses.append([var(r, c, int(ch) - 1)])

    with open("sudoku.cnf", "w") as f:
        f.write(f"p cnf 729 {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(map(str, cl)) + " 0\n")
    print("done")

if __name__ == "__main__":
    main()