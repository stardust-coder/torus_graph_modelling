# Notes on inconsistent notation between Matsuda et al. (2021) and Klein et al.(2020)

## Summary

For torus graphical models, the score matching objective used in the torus graph literature is scaled differently from the score matching loss used in Matsuda et al. for SMIC. In particular, the torus-graph score matching loss is exactly one half of Matsuda's convention. Therefore:

- if one wants numerical agreement with the SMIC convention of Matsuda et al., one should multiply the entire torus-scale criterion by \(2\);
- if one only cares about model comparison, this overall factor of \(2\) is irrelevant, since multiplying all candidate scores by the same positive constant does not change the minimizer.

However, this scaling issue is separate from another implementation issue: for each candidate model \(S\), the SMIC should be evaluated using the *unpenalized refitted estimator on \(S\)*, not merely the restriction of the full-model estimator.

---

## 1. SMIC convention

In Matsuda et al., the one-sample score matching loss is defined as

$$
\rho_{\mathrm{SM}}(x,\theta)
=
2\sum_{i=1}^d \frac{\partial^2}{\partial x_i^2}\log \tilde p(x\mid\theta)
+
\sum_{i=1}^d
\left(
\frac{\partial}{\partial x_i}\log \tilde p(x\mid\theta)
\right)^2.
$$

The empirical loss is the sample average

$$
\hat d_{\mathrm{SM}}(\theta)
=
\frac{1}{N}\sum_{t=1}^N \rho_{\mathrm{SM}}(x^{(t)},\theta).
$$

Then SMIC is

$$
\mathrm{SMIC}
=
N\,\hat d_{\mathrm{SM}}(\hat\theta)
+
\operatorname{tr}(\hat I \hat J^{-1}).
$$

For an exponential-family model, Matsuda et al. write the one-sample loss in quadratic form as

$$
\rho_{\mathrm{SM}}(x,\theta)
=
\frac12 \theta^\top \Gamma(x)\theta + g(x)^\top \theta + c(x).
$$

Hence

$$
\hat d_{\mathrm{SM}}(\theta)
=
\frac12 \theta^\top \hat\Gamma \theta + \hat g^\top \theta + \hat c.
$$

---

## 2. Torus graph convention

In the torus graph formulation, the score matching objective is written as

$$
J(\phi)
=
C + \mathbb{E}\!\left[
\frac12 \|\psi(x;\phi)\|^2
+
\sum_{i=1}^d \frac{\partial}{\partial x_i}\psi_i(x;\phi)
\right],
$$

where

$$
\psi(x;\phi)=\nabla_x \log q(x;\phi).
$$

Since \(\log q(x;\phi)=\phi^\top S(x)\), one has

$$
\psi(x;\phi)=D(x)^\top \phi,
\qquad
\frac12 \|\psi(x;\phi)\|^2
=
\frac12 \phi^\top D(x)D(x)^\top \phi.
$$

If we define

$$
\Gamma(x)=D(x)D(x)^\top,
$$

and write the divergence term as

$$
\sum_{i=1}^d \frac{\partial}{\partial x_i}\psi_i(x;\phi)
=
-\phi^\top H(x),
$$

then the one-sample torus loss is

$$
\rho_{\mathrm{torus}}(x,\phi)
=
\frac12 \phi^\top \Gamma(x)\phi - \phi^\top H(x).
$$

Thus the empirical torus objective is

$$
\hat d_{\mathrm{torus}}(\phi)
=
\frac12 \phi^\top \hat\Gamma \phi - \phi^\top \hat H + \text{const.}
$$

---

## 3. Relation between the two conventions

Comparing the two definitions, the torus-graph loss is exactly one half of Matsuda's score matching loss:

$$
\rho_{\mathrm{SM}}^{\text{(Matsuda)}}(x,\phi)
=
2\,\rho_{\mathrm{torus}}(x,\phi).
$$

Therefore

$$
\hat d_{\mathrm{SM}}^{\text{(Matsuda)}}(\phi)
=
2\,\hat d_{\mathrm{torus}}(\phi),
$$

up to the same additive constant convention.

So if one computes an information criterion using the torus-scale loss, then the Matsuda-scale version is obtained by multiplying the *entire* criterion by \(2\).

---

## 4. Why the correction term also scales by 2

Suppose the loss is multiplied by a positive constant \(c\). Then:

- the estimating function scales by \(c\),
- the outer-product matrix \(\hat I\) scales by \(c^2\),
- the Hessian-type matrix \(\hat J\) scales by \(c\).

Hence

$$
\operatorname{tr}(\hat I' \hat J'^{-1})
=
\operatorname{tr}\!\bigl((c^2\hat I)(c\hat J)^{-1}\bigr)
=
c\,\operatorname{tr}(\hat I \hat J^{-1}).
$$

Therefore the full criterion scales as

$$
\mathrm{SMIC}' = c\,\mathrm{SMIC}.
$$

In the present case \(c=2\), so

$$
\mathrm{SMIC}_{\text{Matsuda-scale}}
=
2\,\mathrm{SMIC}_{\text{torus-scale}}.
$$

---

## 5. Consequence for model selection

If the only goal is model comparison, then multiplying every candidate score by the same positive constant does not matter:

$$
\arg\min_S \mathrm{SMIC}_{\text{Matsuda-scale}}(S)
=
\arg\min_S \mathrm{SMIC}_{\text{torus-scale}}(S).
$$

So the final factor of 2 is irrelevant for choosing the best model.

