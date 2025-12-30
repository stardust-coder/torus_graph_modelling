#いったんデータ整形
from misc import numpy_to_latex_table
import pandas as pd
import ast  # eval より安全な literal_eval を推奨

def is_full_rank(A, tol=None):
    """
    NumPy の行列 A が full-rank かどうかを判定する。
    
    Parameters
    ----------
    A : ndarray
        チェックする行列
    tol : float or None
        特異値分解の閾値（None なら NumPy のデフォルト）
    
    Returns
    -------
    bool
        True なら full rank、False なら rank deficient（欠陥）
    """
    A = np.asarray(A)
    r = np.linalg.matrix_rank(A, tol=tol)
    return r == min(A.shape)

def load_df(path):
    # ===== 1. CSV 読み込み =====
    # あなたのファイル名に合わせてパスを変えてください
    df_wide = pd.read_csv(path, sep="\t")

    # 期待している列：
    # patient_ID, group, baseline, mild, moderate, recovery
    # print(df_wide.head())

    # ===== 2. "('640',0.405,...)" みたいなテキストをタプルに変換 =====
    tuple_cols = ["baseline", "mild", "moderate", "recovery"]

    for col in tuple_cols:
        df_wide[col] = df_wide[col].apply(ast.literal_eval)
        # eval(col) を使うなら ast.literal_eval の代わりに eval でもよいが、自己責任で…

    # print(df_wide[tuple_cols].head())

    # ===== 3. wide → long に変形 =====
    df_long = (
        df_wide
        .set_index(["patient_ID", "group"])
        .stack()  # baseline/mild/... を縦方向に並べる
        .reset_index()
        .rename(columns={"level_2": "state", 0: "vals"})
    )

    # vals のタプルを4列に分解
    df_long[["count", "v2", "v3", "v4"]] = pd.DataFrame(
        df_long["vals"].tolist(), index=df_long.index
    )

    df_long = df_long.drop(columns=["vals"])

    # 型をきれいにしておく
    df_long["patient_ID"] = df_long["patient_ID"].astype("category")
    df_long["group"] = df_long["group"].astype("category")
    df_long["state"] = df_long["state"].astype("category")

    def anscombe(x):
        from math import sqrt
        return 2*sqrt(x+3/8)

    df_long["v1"] = df_long["count"].apply(anscombe)
    return df_long

import numpy as np
import pandas as pd
from scipy.stats import chi2

def _effect_code_levels(cat_series):
    """
    カテゴリ変数を sum-to-zero (effect coding) で表現する。

    L 水準のカテゴリに対して (L-1) 列の行列 C を返す。

    コーディング規則:
        - 列 k は、レベル k に 1、最終レベル L-1 に -1、その他 0。
    これにより、各列について「全レベルでの和 = 0」となり、
    パラメータに Σ α_l = 0 の制約が入ったことに相当する。
    """
    cat = cat_series.astype("category")
    levels = list(cat.cat.categories)
    L = len(levels)
    n = len(cat)

    if L <= 1:
        # 1 水準なら効果は識別不能なので何も返さない
        return np.empty((n, 0)), []

    cols = []
    names = []
    base_level = levels[-1]

    for lev in levels[:-1]:
        col = np.zeros(n, dtype=float)
        col[cat == lev] = 1.0
        col[cat == base_level] = -1.0
        cols.append(col.reshape(-1, 1))
        names.append(f"{cat_series.name}[{lev}]-{base_level}")

    X = np.hstack(cols)
    return X, names


def _build_design_matrix(dfw,
                         include_group=True,
                         include_state=True,
                         include_inter=True):
    """
    和の制約付き (sum-to-zero) パラメータ化に対応したデザイン行列を構築する。

    モデル:
        y = μ
            + α_g (Σ_g α_g = 0)
            + β_s (Σ_s β_s = 0)
            + (αβ)_{gs} (Σ_g (αβ)_{gs} = 0, Σ_s (αβ)_{gs} = 0)
            + γ_{g,subj} (各 group ごとに Σ_subj γ_{g,subj} = 0)
            + 誤差

    include_* フラグで group, state, interaction の項を入/切り替え。
    subject-within-group（個体効果）は常に入れる。
    """
    n = len(dfw)

    group = dfw["group"].astype("category")
    state = dfw["state"].astype("category")
    subject = dfw["subject"].astype("category")

    group_levels = list(group.cat.categories)
    state_levels = list(state.cat.categories)

    cols = []
    names = []

    # 1) 切片（全体平均 μ）
    cols.append(np.ones((n, 1), dtype=float))
    names.append("Intercept")

    # 2) group の主効果: sum-to-zero effect coding
    if include_group:
        Xg, names_g = _effect_code_levels(group)
        cols.append(Xg)
        names.extend([f"group:{nm}" for nm in names_g])
        group_contrast_cols = Xg
    else:
        group_contrast_cols = np.empty((n, 0))

    # 3) state の主効果: sum-to-zero effect coding
    if include_state:
        Xs, names_s = _effect_code_levels(state)
        cols.append(Xs)
        names.extend([f"state:{nm}" for nm in names_s])
        state_contrast_cols = Xs
    else:
        state_contrast_cols = np.empty((n, 0))

    # 4) interaction: group × state の効果
    #    ここも sum-to-zero コーディングされた group/state を掛け合わせることで
    #    Σ_i (αβ)_{ij} = 0, Σ_j (αβ)_{ij} = 0 の制約が入った形になる
    if include_inter and include_group and include_state:
        inter_cols = []
        inter_names = []
        for gi in range(group_contrast_cols.shape[1]):
            for sj in range(state_contrast_cols.shape[1]):
                col = (group_contrast_cols[:, gi] *
                       state_contrast_cols[:, sj]).reshape(-1, 1)
                inter_cols.append(col)
                inter_names.append(
                    f"inter:g{gi}*s{sj}"
                )
        if inter_cols:
            Xint = np.hstack(inter_cols)
            cols.append(Xint)
            names.extend(inter_names)

    # 5) subject-within-group（個体効果）
    #    各 group g について、その group に属する subject を列挙し、
    #    n_g 個のうち (n_g - 1) 個をパラメータとして持ち、
    #    最後の subject の効果を
    #       γ_{g,last} = - Σ_{others} γ_{g,subj}
    #    になるように効果コーディングする。
    subj_cols = []
    subj_names = []

    for g in group_levels:
        mask_g = (group == g)
        subj_in_g = list(dfw.loc[mask_g, "subject"].unique())

        if len(subj_in_g) <= 1:
            # 1 人だけなら制約で効果は 0 なので列を作らない
            continue

        base_subj = subj_in_g[-1]

        for s in subj_in_g[:-1]:
            col = np.zeros(n, dtype=float)

            # その group の subject=s に 1、subject=base_subj に -1
            mask_sub = mask_g & (subject == s)
            mask_base = mask_g & (subject == base_subj)

            col[mask_sub] = 1.0
            col[mask_base] = -1.0

            subj_cols.append(col.reshape(-1, 1))
            subj_names.append(f"sub({g})[{s}]-{base_subj}")

    if subj_cols:
        Xsubj = np.hstack(subj_cols)
        cols.append(Xsubj)
        names.extend(subj_names)

    X = np.hstack(cols)
    return X, names



def _fit_ols(X, y):
    """
    OLS を自前で解き、正規誤差を仮定した対数尤度を計算。
    """
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    if residuals.size > 0:
        RSS = residuals[0]
    else:
        RSS = np.sum((y - X @ beta) ** 2)

    n = len(y)
    sigma2 = RSS / n
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    df_model = X.shape[1]

    return {
        "beta": beta,
        "RSS": RSS,
        "sigma2": sigma2,
        "ll": ll,
        "df_model": df_model,
    }


def lrt_anova_two_way_repeated(df, y_col, group_col, state_col, subject_col):
    """
    Two-way mixed-design (group × state) について、
    和の制約付きパラメータ化を使った線形モデルによる LRT を行う。

    制約:
        Σ_g α_g = 0
        Σ_s β_s = 0
        Σ_i (αβ)_{ij} = 0
        Σ_j (αβ)_{ij} = 0
        各 group ごとに Σ_subj γ_{g,subj} = 0

    ここでは subject 効果を「固定効果として」入れた上で、
    group, state, group×state の効果に対して尤度比検定を行う。
    """

    dfw = df.copy()
    dfw = dfw.rename(columns={
        y_col: "y",
        group_col: "group",
        state_col: "state",
        subject_col: "subject"
    })

    dfw["group"] = dfw["group"].astype("category")
    dfw["state"] = dfw["state"].astype("category")
    dfw["subject"] = dfw["subject"].astype("category")

    y = dfw["y"].to_numpy(dtype=float)
    results = []

    # -------------------------------------------------------
    # FULL MODEL : y ~ group * state + subject-within-group
    # -------------------------------------------------------
    X_full, X_full_names = _build_design_matrix(
        dfw,
        include_group=True,
        include_state=True,
        include_inter=True
    )
    full = _fit_ols(X_full, y)
    ll_full = full["ll"]
    df_full = full["df_model"]
    # import pdb; pdb.set_trace()

    # -------------------------------------------------------
    # 1) TEST INTERACTION : drop group:state
    # -------------------------------------------------------
    X_no_inter, _ = _build_design_matrix(
        dfw,
        include_group=True,
        include_state=True,
        include_inter=False
    )
    no_inter = _fit_ols(X_no_inter, y)
    ll_no_inter = no_inter["ll"]
    df_no_inter = no_inter["df_model"]

    LR_inter = 2 * (ll_full - ll_no_inter)
    df_inter = df_full - df_no_inter
    p_inter = chi2.sf(LR_inter, df_inter)

    results.append({
        "effect": "group:state",
        "LR": LR_inter,
        "df": df_inter,
        "p_value": p_inter,
        "ll_full": ll_full,
        "ll_reduced": ll_no_inter
    })

    # import pdb; pdb.set_trace()
    # -------------------------------------------------------
    # 2) TEST GROUP : drop group （subject-within-group は残す）
    #    full と reduced でどちらも interaction を含まないようにして、
    #    「group 主効果の有無」の比較にする。
    # -------------------------------------------------------
    X_no_group, _ = _build_design_matrix(
        dfw,
        include_group=False,
        include_state=True,
        include_inter=False
    )
    no_group = _fit_ols(X_no_group, y)
    ll_no_group = no_group["ll"]
    df_no_group = no_group["df_model"]

    LR_group = 2 * (ll_full - ll_no_group)
    df_group = df_full - df_no_group
    p_group = chi2.sf(LR_group, df_group)

    results.append({
        "effect": "group",
        "LR": LR_group,
        "df": df_group,
        "p_value": p_group,
        "ll_full": ll_full,
        "ll_reduced": ll_no_group
    })

    # import pdb; pdb.set_trace()

    # -------------------------------------------------------
    # 3) TEST STATE : drop state
    #    （group, subject-within-group は残す）
    # -------------------------------------------------------
    X_no_state, _ = _build_design_matrix(
        dfw,
        include_group=True,
        include_state=False,
        include_inter=False
    )
    
    no_state = _fit_ols(X_no_state, y)
    ll_no_state = no_state["ll"]
    df_no_state = no_state["df_model"]

    LR_state = 2 * (ll_full - ll_no_state)
    df_state = df_full - df_no_state
    p_state = chi2.sf(LR_state, df_state)

    results.append({
        "effect": "state",
        "LR": LR_state,
        "df": df_state,
        "p_value": p_state,
        "ll_full": ll_full,
        "ll_reduced": ll_no_state
    })

    # import pdb; pdb.set_trace()

    return pd.DataFrame(results).set_index("effect")


df_long = load_df("/home/sukeda/torus_graph_modelling/output/HumanEEG/alpha.tsv")
latex = []
for i in range(1,5):
    anova_vi = lrt_anova_two_way_repeated(
        df=df_long,
        y_col=f"v{i}",
        group_col="group",
        state_col="state",
        subject_col="patient_ID"
    )
    latex.append(anova_vi["p_value"].tolist())
    print(anova_vi)

print(numpy_to_latex_table(np.array(latex).T))