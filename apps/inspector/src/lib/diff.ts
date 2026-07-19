// Minimal line-level LCS diff for persona-card rewrites. No dependency —
// this is a small, well-understood algorithm and pulling in a diff library
// for a handful of short string arrays isn't worth it.

export type DiffOp = { type: "unchanged" | "removed" | "added"; text: string };

/** Longest-common-subsequence line diff between two string arrays. */
export function lineDiff(before: string[], after: string[]): DiffOp[] {
  const n = before.length;
  const m = after.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));

  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] =
        before[i] === after[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }

  const ops: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (before[i] === after[j]) {
      ops.push({ type: "unchanged", text: before[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      ops.push({ type: "removed", text: before[i] });
      i++;
    } else {
      ops.push({ type: "added", text: after[j] });
      j++;
    }
  }
  while (i < n) {
    ops.push({ type: "removed", text: before[i] });
    i++;
  }
  while (j < m) {
    ops.push({ type: "added", text: after[j] });
    j++;
  }
  return ops;
}
