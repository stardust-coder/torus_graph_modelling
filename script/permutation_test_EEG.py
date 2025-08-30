# -*- coding: utf-8 -*-
"""
TSVファイルに対して permutation test を実行し、
metric_idx=0〜3の p値を LaTeX table 形式でまとめて出力するコード
"""

import re
import math
import itertools
from typing import Optional, Tuple
import pandas as pd
import numpy as np


# ====== データ処理ユーティリティ ======
TUP_RE = re.compile(r"\(\s*([^\)]+)\s*\)")

def parse_tuple_str(s: str) -> Tuple[float, float, float, float]:
    """"(a,b,c,d)" を float のタプルに変換"""
    if pd.isna(s):
        return (math.nan,)*4
    m = TUP_RE.search(str(s))
    parts = [p.strip() for p in m.group(1).split(",")]
    return tuple(float(x) for x in parts)

def tidy_from_file(path: str) -> pd.DataFrame:
    """TSVを tidy 形式に変換"""
    df = pd.read_csv(path, sep="\t")
    rows = []
    for _, row in df.iterrows():
        for phase in ["baseline","mild","moderate","recovery"]:
            a,b,c,d = parse_tuple_str(row[phase])
            for k,v in enumerate([a,b,c,d]):
                rows.append({
                    "patient_ID": int(row["patient_ID"]),
                    "group": str(row["group"]),
                    "phase": phase,
                    "metric_idx": k,
                    "value": float(v)
                })
    return pd.DataFrame(rows)


# ====== permutation test ======
def perm_test_independent(x: np.ndarray, y: np.ndarray, n_perm=10000, random_state: Optional[int]=0) -> Tuple[float,float]:
    """独立2群 permutation test: 平均差とp値"""
    rng = np.random.default_rng(random_state)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    obs = np.mean(x) - np.mean(y)
    pooled = np.concatenate([x,y])
    n_x = len(x)
    count=0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        stat = np.mean(pooled[:n_x]) - np.mean(pooled[n_x:])
        if abs(stat) >= abs(obs):
            count+=1
    p = (count+1)/(n_perm+1)
    return obs,p

def perm_test_paired(a: np.ndarray, b: np.ndarray, n_perm=10000, random_state: Optional[int]=0) -> Tuple[float,float]:
    """対応あり permutation test: 差の平均とp値"""
    rng = np.random.default_rng(random_state)
    mask = ~np.isnan(a)&~np.isnan(b)
    d = (a[mask]-b[mask]).astype(float)
    obs = np.mean(d)
    n=len(d)
    count=0
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=n)
        stat = np.mean(signs*d)
        if abs(stat)>=abs(obs):
            count+=1
    p = (count+1)/(n_perm+1)
    return obs,p


# ====== ヘルパ ======
def select_by_group_phase(tidy: pd.DataFrame, group: str, phase: str, metric_idx: int) -> np.ndarray:
    return tidy.query("group==@group and phase==@phase and metric_idx==@metric_idx")["value"].to_numpy()

def paired_values_by_phase(tidy: pd.DataFrame, phase_a: str, phase_b: str, metric_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    df = tidy.query("metric_idx==@metric_idx")
    a = df[df["phase"]==phase_a].set_index("patient_ID")["value"]
    b = df[df["phase"]==phase_b].set_index("patient_ID")["value"]
    common = a.index.intersection(b.index)
    return a.loc[common].to_numpy(), b.loc[common].to_numpy()


# ====== メイン処理 ======
def make_result_tables_tex_p_all_parentheses(path: str):
    tidy = tidy_from_file(path)
    phases = ["baseline","mild","moderate","recovery"]

    # ---- 表1: フェーズ間（対応あり, p値のみ metric0-3を括弧で括る）----
    mat = pd.DataFrame("", index=phases, columns=phases, dtype=object)
    for i, j in itertools.combinations(phases, 2):
        pvals = []
        for m in [0,1,2,3]:
            a, b = paired_values_by_phase(tidy, i, j, m)
            _, p = perm_test_paired(a, b)
            pvals.append(f"{p:.3f}")
        mat.loc[i,j] = "(" + ", ".join(pvals) + ")"

    # ---- 表2: drowsy vs responsive（独立2群, p値のみ metric0-3を括る）----
    rows = {}
    for ph in phases:
        pvals = []
        for m in [0,1,2,3]:
            x = select_by_group_phase(tidy,"drowsy",ph,m)
            y = select_by_group_phase(tidy,"responsive",ph,m)
            _, p = perm_test_independent(x,y)
            pvals.append(f"{p:.3f}")
        rows[ph] = "(" + ", ".join(pvals) + ")"
    df2 = pd.DataFrame.from_dict(rows,orient="index",columns=["drowsy vs responsive"])

    # ---- LaTeX 出力 ----
    tex1 = "\\begin{table}[htbp]\n\\centering\n"
    tex1 += "\\caption{Phase間の対応ありPermutation Test (p値, metric0-3)}\n"
    tex1 += "\\begin{tabular}{lcccc}\n\\hline\n"
    tex1 += " & " + " & ".join(phases) + " \\\\\n\\hline\n"
    for i in phases:
        row = [i]
        for j in phases:
            if i==j:
                row.append("")
            elif mat.loc[i,j]!="":
                row.append(mat.loc[i,j])
            else:
                row.append("")
        tex1 += " & ".join(row) + " \\\\\n"
    tex1 += "\\hline\n\\end{tabular}\n\\end{table}\n"

    tex2 = "\\begin{table}[htbp]\n\\centering\n"
    tex2 += "\\caption{drowsy vs responsive (p値, metric0-3)}\n"
    tex2 += "\\begin{tabular}{lc}\n\\hline\nPhase & drowsy vs responsive \\\\\n\\hline\n"
    for ph in phases:
        tex2 += f"{ph} & {df2.loc[ph,'drowsy vs responsive']} \\\\\n"
    tex2 += "\\hline\n\\end{tabular}\n\\end{table}\n"

    return tex1, tex2


# ====== 実行例 ======
if __name__=="__main__":
    # ファイルパスを指定
    path = "output/HumanEEG/gamma.tsv"   # ← あなたのTSVファイル名に変更してください
    tex1, tex2 = make_result_tables_tex_p_all_parentheses(path)

    print("=== LaTeX Table 1 (Phase間) ===")
    print(tex1)
    print("\n=== LaTeX Table 2 (Group間) ===")
    print(tex2)
