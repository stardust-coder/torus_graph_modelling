import numpy as np

def truncate(x, decimals=3):
    """小数点以下を切り捨てる関数"""
    factor = 10 ** decimals
    return np.trunc(x * factor) / factor

def add_bold(x,text):
    if x < 0.05:
        return f"\\textbf{{{text}}}"
    else:
        return text

def numpy_to_latex_table(arr, decimals=3, env=True):
    """
    arr : 2D numpy array
    decimals : 切り捨て小数点以下桁数
    env : tabular 環境を付けるかどうか
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")

    rows = []
    for row in arr:
        cells = [add_bold(x,f"{truncate(x, decimals):.{decimals}f}") for x in row]
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)

    if not env:
        return body

    cols = "c" * arr.shape[1]
    latex = (
        f"\\begin{{tabular}}{{{cols}}}\n"
        + body +
        "\n\\end{tabular}"
    )
    return latex

