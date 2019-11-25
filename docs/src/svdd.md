# [SVDD](@id svdd_doc)

Support Vector Data Description (SVDD) is a one-class classifier and was first presented in [1].

This documentation first presents the optimization problem and then guides the guides the reader to the dual problem

## SVDD Overview

SVDD is an optimization problem of the following form.

```math
  \begin{aligned}
  P: \ & \underset{R, a, \xi}{\text{min}}
  & & R^2 + C * \sum_i \xi_i  \\
  & \text{s.t.}
  & & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \leq R^2 + \xi_i, \; ∀ i \\
  & & & \xi_i \geq 0, \; ∀ i
  \end{aligned}
```
with radius $R$ and center of the hypersphere $a$, a slack variable $\xi$, and a mapping into an implicit feature space $\Phi$.

The dual Problem of the Lagrangian is:

```math
\begin{aligned}
D: \ & \underset{\alpha}{\text{max}}
& & \sum_{i}\alpha_i K_{i,i} - \sum_i\sum_j\alpha_i\alpha_jK_{i,j}  \\
& \text{s.t.}
& & \sum_i \alpha_i = 1 \\
& & & 0 \leq \alpha_i \leq C, \; ∀ i \\
\end{aligned}
```
where $\alpha$ are the Lagrange multipliers, and $K_{i,j} = \langle {\Phi(x_i),\Phi{x_j}} \rangle$ the inner product in the implicit feature space.
Solving the Lagrangian gives an optimal $α$. The following rules are valid for the optimal $α$:
```math
\begin{aligned}
\text{(I)} \quad & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \lt R^2 & ⇒ &\quad α_i = 0\\
\text{(II)} \quad & \left\Vert \Phi(x_{i}) - a \right\Vert^2 = R^2 & ⇒ &\quad 0 < α_i < C\\
\text{(III)} \quad & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \gt R^2 & ⇒ &\quad α_i = C
\end{aligned}
```
An observation is an inlier if (I) or (II) holds and an outlier if (III) holds. Additionally, an observation is called a support vector (SV) if (II) holds. One can calculate the radius of the hypersphere from any SV as we will explain later.

## Derivation of the Dual Problem

#### Preliminaries on Lagrangian duals.

Given a basic minimization problem with a objective function $f(x)$, inequality constraints $g_i(x)$ and equality constraints $h_j(x)$:
```math
  \begin{aligned}
  P: \ & \underset{x}{\text{min}}
  & & f(x)  \\
  & \text{s.t.}
  & & g_i(x) \leq 0, \; ∀ i\\
  & & & h_j(x) = 0, \; ∀ j
  \end{aligned}
```
The Lagrangian is of $P$ is $\mathcal{L}(x, α, β) = f(x) + \sum_i α_i g_i(x) + \sum_j β_j h_j(x)$ with dual variables $α_i \geq 0$ and $β_j \geq 0$.

Instead of solving the primal problem $P$, one can then solve Wolfe dual problem based on the Lagrangian when the functions $f$, $g_i$ and $h_j$ are differentiable and convex. Note that Dual problem is now an maximization problem.
```math
\begin{aligned}
D: \ & \underset{x, α, β}{\text{max}}
& & f(x) + \sum_i α_i g_i(x) + \sum_j β_j h_j(x)  \\
& \text{s.t.}
& & ∇f(x) + \sum_i α_i ∇ g_i(x) + \sum_j β_j ∇ h_j(x) = 0\\
& & & α_i \geq 0, \; ∀ i \\
& & & β_j \geq 0, \; ∀ j \\
\end{aligned}
```

#### SVDD

Given the primal optimization problem of the SVDD
```math
  \begin{aligned}
  P: \ & \underset{R, a, \xi}{\text{min}}
  & & R^2 + C * \sum_i \xi_i  \\
  & \text{s.t.}
  & & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \leq R^2 + \xi_i, \; ∀ i \\
  & & & \xi_i \geq 0, \; ∀ i
  \end{aligned}
```
we transpose the constraints to $\left\Vert \Phi(x_{i}) - a \right\Vert^2 - R^2 - \xi_i \leq 0$ and $-\xi_i \leq 0$. Then the Lagrangian is
```math
\begin{aligned}
\mathcal{L}(R, a, α, γ, ξ) &= R^2 + C \sum_i \xi_i + \sum_i α_i (\left\Vert \Phi(x_{i}) - a \right\Vert^2 - R^2 - \xi_i) + \sum_i γ_i (-\xi_i) \\
&= R^2 + C \sum_i \xi_i - \sum_i α_i ( R^2 + \xi_i - \left\Vert \Phi(x_{i}) - a \right\Vert^2) - \sum_i γ_i \xi_i \\
&= R^2 + C \sum_i \xi_i - \sum_i α_i ( R^2 + \xi_i - \left\Vert \Phi(x_{i})\right\Vert^2 - 2\Phi(x_i) a + \left\Vert a\right\Vert^2) - \sum_i γ_i \xi_i
\end{aligned}
```
with dual variables $α_i \geq 0$ and $γ_i \geq 0$.

Setting the partial derivatives of $\mathcal{L}$ to zero gives
```math
\begin{aligned}
\dfrac{\partial \mathcal{L}}{\partial R} = 0:\quad&
2R - 2\sum_i α_i R &= 0\\
⇔ \quad& 2R (1 - \sum_i α_i) &= 0\\
⇔ \quad& \sum_i α_i = 1 \quad \text{with } R > 0\\
\dfrac{\partial \mathcal{L}}{\partial a} = 0:\quad&
- \sum \alpha_i(2a - 2 \Phi(x_i)) &= 0\\
⇔ \quad& - a\sum_i \alpha_i  + \sum_i \alpha_i\Phi(x_i) &= 0\\
⇔ \quad& a = \dfrac{\sum_i \alpha_i \Phi(x_i)}{\sum_i \alpha_i} = \sum_i \alpha_i \Phi(x_i)\\
\dfrac{\partial \mathcal{L}}{\partial \xi_i} = 0:\quad&
c - \alpha_i - \gamma_i &= 0&\\
⇔ \quad& \alpha_i = C - \gamma_i&\\
\Rightarrow \quad& 0 \leq \alpha_i \leq C \quad \text{with } \alpha_i, \gamma_i \geq 0\\
\end{aligned}
```

Substituting back into $\mathcal{L}$ gives
```math
\begin{aligned}
\mathcal{L}(R, a, α, γ, ξ) &= R^2 + C \sum_i \xi_i - \sum_i α_i \left( R^2 + \xi_i - \left\Vert \Phi(x_{i})\right\Vert^2 - 2\Phi(x_i) a + \left\Vert a\right\Vert^2\right) - \sum_i γ_i \xi_i\\
&= R^2(1 - \underbrace{\sum_i \alpha_i}_{=1}) + \sum_i \xi_i \underbrace{(C - \alpha_i - \gamma_i)}_{=0} + \sum_i \alpha_i (\left\Vert \Phi(x_{i})\right\Vert^2 - 2\Phi(x_i) a + \left\Vert a\right\Vert^2)\\
&= \sum_i \alpha_i (\left\Vert \Phi(x_{i})\right\Vert^2 - 2\Phi(x_i) a + \left\Vert a\right\Vert^2)\\
&= \sum_i \alpha_i \left(\left\Vert \Phi(x_{i})\right\Vert^2 - 2 \Phi(x_i)\sum_j \alpha_j \Phi(x_j) + \left(\sum_j \alpha_j \Phi(x_j)\right)^2 \right)\\
&= \sum_i \alpha_i \left(\left\Vert \Phi(x_{i})\right\Vert^2 - 2 \Phi(x_i)\sum_j \alpha_j \Phi(x_j) + \sum_j \sum_k \alpha_j \alpha_k \Phi(x_j)\Phi(x_k) \right)\\
&=\sum_i \alpha_i\left\Vert \Phi(x_{i})\right\Vert^2 - 2 \sum_i\sum_j \alpha_i \alpha_j \Phi(x_i) \Phi(x_j) + \sum_i\sum_j\sum_k\alpha_i\alpha_j\alpha_k\Phi(x_i)\Phi(x_j)\Phi(x_k)\\
&=\sum_i \alpha_i\left\Vert \Phi(x_{i})\right\Vert^2 - \sum_i\sum_j \alpha_i \alpha_j \Phi(x_i) \Phi(x_j) (2 - \underbrace{\sum_k \alpha_k \Phi(x_k)}_{=1})\\
&=\sum_i \alpha_i\left\Vert \Phi(x_{i})\right\Vert^2 - \sum_i\sum_j \alpha_i \alpha_j \Phi(x_i) \Phi(x_j)\\
&=\sum_i \alpha_i\Phi(x_{i})\Phi(x_i) - \sum_i\sum_j \alpha_i \alpha_j \Phi(x_i) \Phi(x_j)
\end{aligned}
```
By substituting the inner products with the kernel matrix $K_{i, j} = \Phi(x_i)\Phi(x_j)$ and adding the constraints we finally get the dual problem:
```math
\begin{aligned}
D: \ & \underset{\alpha}{\text{max}}
& & \sum_{i}\alpha_i K_{i,i} - \sum_i\sum_j\alpha_i\alpha_jK_{i,j}  \\
& \text{s.t.}
& & \sum_i \alpha_i = 1\\
& & & 0 \leq \alpha_i \leq C, \; ∀ i \\
\end{aligned}
```

The decision function of the SVDD is the distance to the decision boundary; positive for outliers, negative or zero for inliers:
```math
f(x_i) = \left\Vert \Phi(x_i) - a \right\Vert^2 - R^2
```

$R^2$ can be calculated with any support vector (SV), i.e., an observation $x_k$ with an $\alpha_k$ that is $0 < \alpha_k < C$:

```math
\begin{aligned}
R^2 &= \left\Vert \Phi(x_k) - a \right\Vert^2\\
& = \Phi(x_k)\Phi(x_k) - 2 \Phi(x_k)\sum_i \alpha_i \Phi(x_i) + \sum_i\sum_j \alpha_i \alpha_j \Phi(x_i) \Phi(x_j)\\
& = K_{k, k} - 2 \sum_i \alpha_i K_{k, i} + \sum_i \sum_j \alpha_i \alpha_j K_{i, j}
\end{aligned}
```

The final decision function for an arbitrary $x_i$ is then:
```math
f(x_i) = K_{i, i} - 2 \sum_j \alpha_j K_{i, j} + \underbrace{\sum_j\sum_k \alpha_j \alpha_k K_{j, k}}_{\text{const}} - R^2
```
The term with the double sum is independent of $x_i$ and can be pre-calculcated.

# SVDDneg

The SVDDneg is an extension of the vanillia SVDD and allows to use outlier labels [1]. In the following the key extension of the SVDD are highlighted in red. The outliers have index $l$ in the following optimization problem:

```math
  \begin{aligned}
  P: \ & \underset{R, a, \xi}{\text{min}}
  & & R^2 + C_1 * \sum_i \xi_i  {\color{red}+ C_2 * \sum_l \xi_l}\\
  & \text{s.t.}
  & & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \leq R^2 + \xi_i, \; ∀ i \\
  & & & {\color{red}\left\Vert \Phi(x_{l}) - a \right\Vert^2 \leq R^2 - \xi_l, \; ∀ l }\\
  & & & \xi_i, \geq 0, \; ∀ i\\
  & & & {\color{red}\xi_l, \geq 0, \; ∀ l}
  \end{aligned}
```
with radius $R$ and center of the hypersphere $a$, a slack variable $\xi$, and a mapping into an implicit feature space $\Phi$.

The dual SVDDneg problem with inlier/unlabeled indices $i$, $j$ and outlier indices $l$, $m$ of the Lagrangian is:

```math
\begin{aligned}
D: \ & \underset{\alpha}{\text{max}}
& & \sum_{i}\alpha_i K_{i,i} - \sum_l K_{l, l} - \sum_i \sum_j \alpha_i \alpha_j K_{i, j}\\
& & &+ 2\sum_l\sum_j \alpha_l \alpha_j K_{l, j} - \sum_l\sum_m\alpha_l\alpha_mK_{l,m}  \\
& \text{s.t.}
& & \sum_i \alpha_i = 1 \\
& & & 0 \leq \alpha_i \leq C_1, \; ∀ i \\
& & & 0 \leq \alpha_i \leq C_2, \; ∀ l \\
\end{aligned}
```
where $\alpha$ are the Lagrange multipliers, and $K_{i,j} = \langle {\Phi(x_i),\Phi{x_j}} \rangle$ the inner product in the implicit feature space.


[1] Tax, David MJ, and Robert PW Duin. "Support vector data description." Machine learning 54.1 (2004): 45-66.
[2] Chang, Wei-Cheng, Ching-Pei Lee, and Chih-Jen Lin. "A revisit to support vector data description." Dept. Comput. Sci., Nat. Taiwan Univ., Taipei, Taiwan, Tech. Rep (2013).
