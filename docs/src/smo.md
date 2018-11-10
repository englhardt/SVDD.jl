# SMO

Sequential Minimal Optimization (SMO) is a decomposition method to solve quadratic optimization problems with a specific structure. The original SMO algorithm by John C. Platt has been proposed for Support Vector Machines (SVM). There are several modifications for other types of support vector machines. This section describes the implementation of SMO for Support Vector Data Description (SVDD) [2].

The implementation of SMO for SVDD bases on an adaption of SMO for one-class classification [3]. Therefore, this documentation focuses on the specific adaptions required for SVDD. The following descriptions assume familarity with the basics of SMO [1] and its adaption to one-class SVM [3], and of SVDD [2].

## SVDD Overview

SVDD is an optimization problem of the following form.

```math
  \begin{aligned}
  P: \ & \underset{R, a, \xi}{\text{minimize}}
  & & R^2 + \sum_i \xi_i  \\
  & \text{subject to}
  & & \left\Vert \Phi(x_{i}) - a \right\Vert^2 \leq R^2 + \xi_i, \; ∀ i \\
  & & & \xi_i \geq 0, \; ∀ i
  \end{aligned}
```
with radius $R$ and center of the hypershpere $a$, a slack variable $\xi$, and a mapping into an implicit feature space $\Phi$

The Lagrangian Dual is:

```math
\begin{aligned}
D: \ & \underset{\alpha}{\text{maximize}}
& & \sum_{i,j}\alpha_i\alpha_j K_{i,j} + \sum_i \alpha_iK_{i,i}  \\
& \text{subject to}
& & \sum_i \alpha_i = 1 \\
& & & 0 \leq \alpha_i \leq C, \; ∀ i \\
\end{aligned}
```
where $\alpha$ are the Lagrange multipliers, and $K_{i,j} = \langle {\Phi(x_i),\Phi{x_j}} \rangle$ the inner product in the implicit feature space.
Solving the Lagrangian gives an optimal $α$.

## SMO for SVDD

The basic idea of SMO is to solve reduced versions of the Lagrangian iteratively.
In each iteration, the reduced version of the Lagrangian consists of only two decision variables, i.e., $\alpha_{i1}$ and $\alpha_{i2}$, while $\alpha_j, j∉\{i1, i2\}$ are fixed.
An iteration of SMO consists of two steps:

**Selection Step:** Select $i1$ and $i2$.
  * The search for a good $i2$ are implemented in [`SVDD.smo`](@ref)
  * There are several heuristics to select $i1$ based on the choice for $i2$. These heuristics are implemented in [`SVDD.examineExample!`](@ref)

**Optimization Step:** Solving the reduced Lagrangian for $\alpha_{i1}$ and $\alpha_{i2}$.
  * Implemented in [`SVDD.takeStep!`](@ref)

The iterative procedure converges to the global optimum.
The following sections give details on both steps.

### Optimization Step: Solving the reduced Lagrangian

The following describes how to infer the optimal solution for a given $\alpha_{i1}$ and $\alpha_{i2}$ analytically.

First, $\alpha_{i1}$ and $\alpha_{i2}$ can only be changed in a limited range.
The reason is that after the optimization step, they still have to obey the constraints of the Lagrangian.
From $\sum_i\alpha_i = 1$, one can infer that $Δ = \alpha_{i1} + \alpha_{i2}$ remains constant for one optimization step.
This is, if we add some value to $\alpha_{i2}$, we must remove the same value from $\alpha_{i1}$.
We also know that $\alpha_{i} \geq 0$ and $\alpha_{i} \leq C$.
From this, one can infer the maximum and minumum value that one can add/substract from $\alpha_{i2}$, i.e., one can calculate the lower and the upper bound:

```math
\begin{aligned}
  L &= max(0, \alpha_{i1} + \alpha_{i2} - C)\\
  H &= min(C, \alpha_{i1} + \alpha_{i2})
\end{aligned}
```

(Note: This is slightly different to the original SMO, as one does not need to discern between different labels $y_i \in \{1,-1\}$.)

Second, the optimal value ``\alpha^*_{i2}`` can be derived analytically by setting the partial derivative of the Lagrangian objective function to 0.

```math
f_{D} = \sum_{i,j} \alpha_i \alpha_j K_{i,j} - \sum_{i}\alpha_{i} K_{i,i} \\
\frac{\delta f_{D}}{\alpha_{i2}} = 0

\iff  \alpha^*_{i2} = \frac{2\Delta(K_{i1,i1} - K_{i1,i2}) + C_1 - C_2 - K_{i1,i1} + K_{i2,i2}}{2K_{i1,i1}+2K_{i2, i2}-4K_{i1, i2}}, \\
\text{where} \ C_k=\alpha_{k}\sum_{j=3}^{N}\alpha_j K_{k,j}
```

The resulting value is _clipped_ to the feasible interval.

```
if α*_i2 > H
    α'_i2 = H
elseif α*_i2 < L
    α'_i2 = L
end
```
where `α'_i2` is the updated value of `α_i2` after the optimization step.
It follows that

```
  α'_i1 = Δ - α'_i2
```

To allow the algorithm to converge, one has to decide on a threshold whether the updates to the alpha values has been significant, i.e., if the difference between the old and the new value is above a specified precision.
The implementation uses the decision rule from the original SMO [1, p.10], i.e., update alpha values only if

```math
\lvert\alpha_{i2} - \alpha'_{i2} \rvert > \text{opt_precision} * (\alpha_{i2} + \alpha'_{i2} + \text{opt_precision})
```

where `opt_precision` is a parameter of the optimization algorithm.
This optimization step is implemented in

```@docs
SVDD.takeStep!
```
### Selection Step: Finding a pair (i1, i2)

To take an optimization step, one has to select i1 and i2 first.
The rationale of SMO is to select indices that are likely to make a large step optimization step.
SMO uses heuristics to first select i2, and then select i1 based on it.

##### Selection of i2

A minimum of $P$ has to obey the [KKT conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions).
The relevant KKT condition here is complementary slackness, i.e.,

```math
  \mu_i g_i(x^*) = 0, \, \forall i
```

with dual variable $\mu$ and inequality conditions $g$.
In other words, either the inequality constraint is fulfilled with equality, i.e., $g_i = 0$, or the Lagrange multiplier is zero, i.e., $\mu_i=0$.
For SVDD, this translates to

```math
  \begin{aligned}
    &\left\lVert a - \phi(x_i) \right\rVert^2 < R^2 \rightarrow \alpha_i = 0 \\
    &\left\lVert a - \phi(x_i) \right\rVert^2 = R^2 \rightarrow  0 < \alpha_i < C\\
    &\left\lVert a - \phi(x_i) \right\rVert^2 > R^2 \rightarrow \alpha_i = C
 \end{aligned}
```
See [2] for details.
The distance to the decision boundary is $\left\lVert a - \phi(x_i) \right\rVert^2 - R^2$ which is negative for observations that lie in the hypershpere.

So to check for KKT violations, one has to calculate the distance of $\phi(x_i)$ from the decision boundary, i.e., the left-hand side of the implications above, and compare it with the the respective $\alpha$ value.
The check for KKT violations is implemented in

```@docs
  SVDD.violates_KKT_condition
```

[`SVDD.smo`](@ref) selects $i2$ by searching for indices that violate the KKT conditions.

```@docs
  SVDD.smo
```

This function conducts two tyes of search.

_First type:_ search over the full data set, and randomly selects one of the violating indices.

_Second type:_ restricted search for violations over the subset where $0 <\alpha_i < C$.
These variables are the non-bounded support vectors $SV_{nb}$.

There is one search of the first type, then multiple searches of the second type.
After each search, $i2$ is selected randomly from one of the violating indices, see

```@docs
SVDD.examine_and_update_predictions!
```

##### Selection of i1

SMO selects $i1$ such that the optimization step is as large as possible.
The idea for selecting $i1$ is as follows.
For $\alpha_{i2} > 0$ and negative distance to decision boundary, alpha may decrease.
So a good $\alpha_{i1}$ is one that is likely to increase in the optimization step, i.e., an index where the distance to the decision boundary is positive, and $\alpha_{i1} = 0$.
The heuristic SMO selects the $i1$ with maximum absolute distance between the distance to the center of $i2$ and the distance to the center of some $i1 \in SV_{nb}$.
(Note that using the distance to the decision boundary is equivalent to using the distance to the center in this step).
This selection heuristic is implemented in

```@docs
  SVDD.second_choice_heuristic
```

In some cases, the selected $i1$ does not lead to a positive optimization step.
In this case, there are two fallback strategies.
First, all other indices in $SV_{nb}$ are selected, in random order, whether they result in a positive optimization step.
Second, if there still is no $i1$ that results in a positive optimization step, all remaining indices are selected.
If none of the fallback strategies works, $i2$ is skipped and added to a blacklist.
The fallback strategies are implemented in

```@docs
  SVDD.examineExample!
```

### Termination

If there are no more KKT violations, the algorithm terminates.

### Further implementation details

This section describes some further implementation details.

##### Initialize alpha

The vector $\alpha$ must be initialized such that it fulfills the constraints of $D$.
The implementation uses the initialization strategy proposed in [3], i.e., randomly setting $\frac{1}{C}$ indices to $C$.
This is implemented in

```@docs
  SVDD.initialize_alpha
```

##### Calculating Distances to Decision Boundary

The distances to the decision boundary are calculated in

```@docs
  SVDD.calculate_predictions
```

In general, to calculate $R$, one can calculate the distance to any non-bounded support vector, i.e., $0 < \alpha_i < C$, as they all lie on the hypershpere.
However, this may not always hold.
There may be cases where the solution for R is not unique, and different support vectors result in different $R$, in particular in intermediate optimization steps where some $\alpha$ values may be non-bounded but violate the KKT conditions.
Therefore, $R$ is averaged over all non-bounded support vectors.
See also [4] for details on non-unique $R$ values.

##### SMO parameters

There are two parameters for SMO: `opt_precision` and `max_iterations`.

`opt_precision` influences the convergence.
Small `opt_precision` values require a larger number of iterations until termination.

`max_iterations` controls the number of times a new $i2$ is selected to attempt an optimization step.

## External API

```@docs
  SVDD.solve!(model::VanillaSVDD, solver::SMOSolver)
```

## References
[1] J. Platt, "Sequential minimal optimization: A fast algorithm for training support vector machines," 1998.

[2] D. M. J. Tax and R. P. W. Duin, "Support Vector Data Description,"" Mach. Learn., 2004.

[3] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, "Estimating the support of a high-dimensional distribution,"" Neural Comput., 2001.

[4] W.-C. Chang, C.-P. Lee, and C.-J. Lin, "A revisit to support vector data description,"Nat. Taiwan Univ., Tech. Rep, 2013.
