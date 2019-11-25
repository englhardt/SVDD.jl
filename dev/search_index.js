var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#SVDD.jl-1",
    "page": "Home",
    "title": "SVDD.jl",
    "category": "section",
    "text": "The documentation for SVDD.jl is still under construction. The examples folder of the repository gives two examples on how to train the different SVDD based classifiers and how to parametrize them.Additional documentation pages:SVDD presents the primal and derivation of the dual optimization problem of the SVDD and SVDDneg\nSMO introduces the SMO heuristic for SVDD that is generally used to solve binary SVM optimization problem"
},

{
    "location": "start/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "start/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": ""
},

{
    "location": "start/#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "SVDD.jl is not yet registered. To install, runpkg> add https://github.com/englhardt/SVDD.jl"
},

{
    "location": "svdd/#",
    "page": "SVDD",
    "title": "SVDD",
    "category": "page",
    "text": ""
},

{
    "location": "svdd/#svdd_doc-1",
    "page": "SVDD",
    "title": "SVDD",
    "category": "section",
    "text": "Support Vector Data Description (SVDD) is a one-class classifier and was first presented in [1].This documentation first presents the optimization problem and then guides the guides the reader to the dual problem"
},

{
    "location": "svdd/#SVDD-Overview-1",
    "page": "SVDD",
    "title": "SVDD Overview",
    "category": "section",
    "text": "SVDD is an optimization problem of the following form.  beginaligned\n  P   undersetR a xitextmin\n    R^2 + C * sum_i xi_i  \n   textst\n    leftVert Phi(x_i) - a rightVert^2 leq R^2 + xi_i   i \n     xi_i geq 0   i\n  endalignedwith radius R and center of the hypersphere a, a slack variable xi, and a mapping into an implicit feature space Phi.The dual Problem of the Lagrangian is:beginaligned\nD   undersetalphatextmax\n  sum_ialpha_i K_ii - sum_isum_jalpha_ialpha_jK_ij  \n textst\n  sum_i alpha_i = 1 \n   0 leq alpha_i leq C   i \nendalignedwhere alpha are the Lagrange multipliers, and K_ij = langle Phi(x_i)Phix_j rangle the inner product in the implicit feature space. Solving the Lagrangian gives an optimal α. The following rules are valid for the optimal α:beginaligned\ntext(I) quad  leftVert Phi(x_i) - a rightVert^2 lt R^2   quad α_i = 0\ntext(II) quad  leftVert Phi(x_i) - a rightVert^2 = R^2   quad 0  α_i  C\ntext(III) quad  leftVert Phi(x_i) - a rightVert^2 gt R^2   quad α_i = C\nendalignedAn observation is an inlier if (I) or (II) holds and an outlier if (III) holds. Additionally, an observation is called a support vector (SV) if (II) holds. One can calculate the radius of the hypersphere from any SV as we will explain later."
},

{
    "location": "svdd/#Derivation-of-the-Dual-Problem-1",
    "page": "SVDD",
    "title": "Derivation of the Dual Problem",
    "category": "section",
    "text": ""
},

{
    "location": "svdd/#Preliminaries-on-Lagrangian-duals.-1",
    "page": "SVDD",
    "title": "Preliminaries on Lagrangian duals.",
    "category": "section",
    "text": "Given a basic minimization problem with a objective function f(x), inequality constraints g_i(x) and equality constraints h_j(x):  beginaligned\n  P   undersetxtextmin\n    f(x)  \n   textst\n    g_i(x) leq 0   i\n     h_j(x) = 0   j\n  endalignedThe Lagrangian is of P is mathcalL(x α β) = f(x) + sum_i α_i g_i(x) + sum_j β_j h_j(x) with dual variables α_i geq 0 and β_j geq 0.Instead of solving the primal problem P, one can then solve Wolfe dual problem based on the Lagrangian when the functions f, g_i and h_j are differentiable and convex. Note that Dual problem is now an maximization problem.beginaligned\nD   undersetx α βtextmax\n  f(x) + sum_i α_i g_i(x) + sum_j β_j h_j(x)  \n textst\n  f(x) + sum_i α_i  g_i(x) + sum_j β_j  h_j(x) = 0\n   α_i geq 0   i \n   β_j geq 0   j \nendaligned"
},

{
    "location": "svdd/#SVDD-1",
    "page": "SVDD",
    "title": "SVDD",
    "category": "section",
    "text": "Given the primal optimization problem of the SVDD  beginaligned\n  P   undersetR a xitextmin\n    R^2 + C * sum_i xi_i  \n   textst\n    leftVert Phi(x_i) - a rightVert^2 leq R^2 + xi_i   i \n     xi_i geq 0   i\n  endalignedwe transpose the constraints to leftVert Phi(x_i) - a rightVert^2 - R^2 - xi_i leq 0 and -xi_i leq 0. Then the Lagrangian isbeginaligned\nmathcalL(R a α γ ξ) = R^2 + C sum_i xi_i + sum_i α_i (leftVert Phi(x_i) - a rightVert^2 - R^2 - xi_i) + sum_i γ_i (-xi_i) \n= R^2 + C sum_i xi_i - sum_i α_i ( R^2 + xi_i - leftVert Phi(x_i) - a rightVert^2) - sum_i γ_i xi_i \n= R^2 + C sum_i xi_i - sum_i α_i ( R^2 + xi_i - leftVert Phi(x_i)rightVert^2 - 2Phi(x_i) a + leftVert arightVert^2) - sum_i γ_i xi_i\nendalignedwith dual variables α_i geq 0 and γ_i geq 0.Setting the partial derivatives of mathcalL to zero givesbeginaligned\ndfracpartial mathcalLpartial R = 0quad\n2R - 2sum_i α_i R = 0\n quad 2R (1 - sum_i α_i) = 0\n quad sum_i α_i = 1 quad textwith  R  0\ndfracpartial mathcalLpartial a = 0quad\n- sum alpha_i(2a - 2 Phi(x_i)) = 0\n quad - asum_i alpha_i  + sum_i alpha_iPhi(x_i) = 0\n quad a = dfracsum_i alpha_i Phi(x_i)sum_i alpha_i = sum_i alpha_i Phi(x_i)\ndfracpartial mathcalLpartial xi_i = 0quad\nc - alpha_i - gamma_i = 0\n quad alpha_i = C - gamma_i\nRightarrow quad 0 leq alpha_i leq C quad textwith  alpha_i gamma_i geq 0\nendalignedSubstituting back into mathcalL givesbeginaligned\nmathcalL(R a α γ ξ) = R^2 + C sum_i xi_i - sum_i α_i left( R^2 + xi_i - leftVert Phi(x_i)rightVert^2 - 2Phi(x_i) a + leftVert arightVert^2right) - sum_i γ_i xi_i\n= R^2(1 - underbracesum_i alpha_i_=1) + sum_i xi_i underbrace(C - alpha_i - gamma_i)_=0 + sum_i alpha_i (leftVert Phi(x_i)rightVert^2 - 2Phi(x_i) a + leftVert arightVert^2)\n= sum_i alpha_i (leftVert Phi(x_i)rightVert^2 - 2Phi(x_i) a + leftVert arightVert^2)\n= sum_i alpha_i left(leftVert Phi(x_i)rightVert^2 - 2 Phi(x_i)sum_j alpha_j Phi(x_j) + left(sum_j alpha_j Phi(x_j)right)^2 right)\n= sum_i alpha_i left(leftVert Phi(x_i)rightVert^2 - 2 Phi(x_i)sum_j alpha_j Phi(x_j) + sum_j sum_k alpha_j alpha_k Phi(x_j)Phi(x_k) right)\n=sum_i alpha_ileftVert Phi(x_i)rightVert^2 - 2 sum_isum_j alpha_i alpha_j Phi(x_i) Phi(x_j) + sum_isum_jsum_kalpha_ialpha_jalpha_kPhi(x_i)Phi(x_j)Phi(x_k)\n=sum_i alpha_ileftVert Phi(x_i)rightVert^2 - sum_isum_j alpha_i alpha_j Phi(x_i) Phi(x_j) (2 - underbracesum_k alpha_k Phi(x_k)_=1)\n=sum_i alpha_ileftVert Phi(x_i)rightVert^2 - sum_isum_j alpha_i alpha_j Phi(x_i) Phi(x_j)\n=sum_i alpha_iPhi(x_i)Phi(x_i) - sum_isum_j alpha_i alpha_j Phi(x_i) Phi(x_j)\nendalignedBy substituting the inner products with the kernel matrix K_i j = Phi(x_i)Phi(x_j) and adding the constraints we finally get the dual problem:beginaligned\nD   undersetalphatextmax\n  sum_ialpha_i K_ii - sum_isum_jalpha_ialpha_jK_ij  \n textst\n  sum_i alpha_i = 1\n   0 leq alpha_i leq C   i \nendalignedThe decision function of the SVDD is the distance to the decision boundary; positive for outliers, negative or zero for inliers:f(x_i) = leftVert Phi(x_i) - a rightVert^2 - R^2R^2can be calculated with any support vector (SV), i.e., an observation x_k with an alpha_k that is 0  alpha_k  C:beginaligned\nR^2 = leftVert Phi(x_k) - a rightVert^2\n = Phi(x_k)Phi(x_k) - 2 Phi(x_k)sum_i alpha_i Phi(x_i) + sum_isum_j alpha_i alpha_j Phi(x_i) Phi(x_j)\n = K_k k - 2 sum_i alpha_i K_k i + sum_i sum_j alpha_i alpha_j K_i j\nendalignedThe final decision function for an arbitrary x_i is then:f(x_i) = K_i i - 2 sum_j alpha_j K_i j + underbracesum_jsum_k alpha_j alpha_k K_j k_textconst - R^2The term with the double sum is independent of x_i and can be pre-calculcated."
},

{
    "location": "svdd/#SVDDneg-1",
    "page": "SVDD",
    "title": "SVDDneg",
    "category": "section",
    "text": "The SVDDneg is an extension of the vanillia SVDD and allows to use outlier labels [1]. In the following the key extension of the SVDD are highlighted in red. The outliers have index l in the following optimization problem:  beginaligned\n  P   undersetR a xitextmin\n    R^2 + C_1 * sum_i xi_i  colorred+ C_2 * sum_l xi_l\n   textst\n    leftVert Phi(x_i) - a rightVert^2 leq R^2 + xi_i   i \n     colorredleftVert Phi(x_l) - a rightVert^2 leq R^2 - xi_l   l \n     xi_i geq 0   i\n     colorredxi_l geq 0   l\n  endalignedwith radius R and center of the hypersphere a, a slack variable xi, and a mapping into an implicit feature space Phi.The dual SVDDneg problem with inlier/unlabeled indices i, j and outlier indices l, m of the Lagrangian is:beginaligned\nD   undersetalphatextmax\n  sum_ialpha_i K_ii - sum_l K_l l - sum_i sum_j alpha_i alpha_j K_i j\n  + 2sum_lsum_j alpha_l alpha_j K_l j - sum_lsum_malpha_lalpha_mK_lm  \n textst\n  sum_i alpha_i = 1 \n   0 leq alpha_i leq C_1   i \n   0 leq alpha_i leq C_2   l \nendalignedwhere alpha are the Lagrange multipliers, and K_ij = langle Phi(x_i)Phix_j rangle the inner product in the implicit feature space.[1] Tax, David MJ, and Robert PW Duin. \"Support vector data description.\" Machine learning 54.1 (2004): 45-66. [2] Chang, Wei-Cheng, Ching-Pei Lee, and Chih-Jen Lin. \"A revisit to support vector data description.\" Dept. Comput. Sci., Nat. Taiwan Univ., Taipei, Taiwan, Tech. Rep (2013)."
},

{
    "location": "smo/#",
    "page": "SMO",
    "title": "SMO",
    "category": "page",
    "text": ""
},

{
    "location": "smo/#smo_doc-1",
    "page": "SMO",
    "title": "SMO",
    "category": "section",
    "text": "Sequential Minimal Optimization (SMO) is a decomposition method to solve quadratic optimization problems with a specific structure. The original SMO algorithm by John C. Platt has been proposed for Support Vector Machines (SVM). There are several modifications for other types of support vector machines. This section describes the implementation of SMO for Support Vector Data Description (SVDD) [2].The implementation of SMO for SVDD bases on an adaption of SMO for one-class classification [3]. Therefore, this documentation focuses on the specific adaptions required for SVDD. The following descriptions assume familarity with the basics of SMO [1] and its adaption to one-class SVM [3], and of SVDD [2]."
},

{
    "location": "smo/#SVDD-Overview-1",
    "page": "SMO",
    "title": "SVDD Overview",
    "category": "section",
    "text": "SVDD is an optimization problem of the following form.  beginaligned\n  P   undersetR a xitextminimize\n    R^2 + sum_i xi_i  \n   textsubject to\n    leftVert Phi(x_i) - a rightVert^2 leq R^2 + xi_i   i \n     xi_i geq 0   i\n  endalignedwith radius R and center of the hypershpere a, a slack variable xi, and a mapping into an implicit feature space PhiThe Lagrangian Dual is:beginaligned\nD   undersetalphatextmaximize\n  sum_i alpha_iK_ii - sum_ijalpha_ialpha_j K_ij \n textsubject to\n  sum_i alpha_i = 1 \n   0 leq alpha_i leq C   i \nendalignedwhere alpha are the Lagrange multipliers, and K_ij = langle Phi(x_i)Phix_j rangle the inner product in the implicit feature space. Solving the Lagrangian gives an optimal α."
},

{
    "location": "smo/#SMO-for-SVDD-1",
    "page": "SMO",
    "title": "SMO for SVDD",
    "category": "section",
    "text": "The basic idea of SMO is to solve reduced versions of the Lagrangian iteratively. In each iteration, the reduced version of the Lagrangian consists of only two decision variables, i.e., alpha_i1 and alpha_i2, while alpha_j ji1 i2 are fixed. An iteration of SMO consists of two steps:Selection Step: Select i1 and i2.The search for a good i2 are implemented in SVDD.smo\nThere are several heuristics to select i1 based on the choice for i2. These heuristics are implemented in SVDD.examineExample!Optimization Step: Solving the reduced Lagrangian for alpha_i1 and alpha_i2.Implemented in SVDD.takeStep!The iterative procedure converges to the global optimum. The following sections give details on both steps."
},

{
    "location": "smo/#SVDD.takeStep!",
    "page": "SMO",
    "title": "SVDD.takeStep!",
    "category": "function",
    "text": "takeStep!(α, i1, i2, K, C, opt_precision)\n\nTake optimization step for i1 and i2 and update α.\n\n\n\n\n\n"
},

{
    "location": "smo/#Optimization-Step:-Solving-the-reduced-Lagrangian-1",
    "page": "SMO",
    "title": "Optimization Step: Solving the reduced Lagrangian",
    "category": "section",
    "text": "The following describes how to infer the optimal solution for a given alpha_i1 and alpha_i2 analytically.First, alpha_i1 and alpha_i2 can only be changed in a limited range. The reason is that after the optimization step, they still have to obey the constraints of the Lagrangian. From sum_ialpha_i = 1, one can infer that Δ = alpha_i1 + alpha_i2 remains constant for one optimization step. This is, if we add some value to alpha_i2, we must remove the same value from alpha_i1. We also know that alpha_i geq 0 and alpha_i leq C. From this, one can infer the maximum and minumum value that one can add/substract from alpha_i2, i.e., one can calculate the lower and the upper bound:beginaligned\n  L = max(0 alpha_i1 + alpha_i2 - C)\n  H = min(C alpha_i1 + alpha_i2)\nendaligned(Note: This is slightly different to the original SMO, as one does not need to discern between different labels y_i in 1-1.)Second, the optimal value alpha^*_i2 can be derived analytically by setting the partial derivative of the Lagrangian objective function to 0.f_D = sum_ij alpha_i alpha_j K_ij - sum_ialpha_i K_ii \nfracdelta f_Dalpha_i2 = 0\n\niff  alpha^*_i2 = frac2Delta(K_i1i1 - K_i1i2) + C_1 - C_2 - K_i1i1 + K_i2i22K_i1i1+2K_i2 i2-4K_i1 i2 \ntextwhere  C_k=alpha_ksum_j=3^Nalpha_j K_kjThe resulting value is clipped to the feasible interval.if α*_i2 > H\n    α\'_i2 = H\nelseif α*_i2 < L\n    α\'_i2 = L\nendwhere α\'_i2 is the updated value of α_i2 after the optimization step. It follows that  α\'_i1 = Δ - α\'_i2To allow the algorithm to converge, one has to decide on a threshold whether the updates to the alpha values has been significant, i.e., if the difference between the old and the new value is above a specified precision. The implementation uses the decision rule from the original SMO [1, p.10], i.e., update alpha values only iflvertalpha_i2 - alpha_i2 rvert  textopt_precision * (alpha_i2 + alpha_i2 + textopt_precision)where opt_precision is a parameter of the optimization algorithm. This optimization step is implemented inSVDD.takeStep!"
},

{
    "location": "smo/#Selection-Step:-Finding-a-pair-(i1,-i2)-1",
    "page": "SMO",
    "title": "Selection Step: Finding a pair (i1, i2)",
    "category": "section",
    "text": "To take an optimization step, one has to select i1 and i2 first. The rationale of SMO is to select indices that are likely to make a large step optimization step. SMO uses heuristics to first select i2, and then select i1 based on it."
},

{
    "location": "smo/#SVDD.violates_KKT_condition",
    "page": "SMO",
    "title": "SVDD.violates_KKT_condition",
    "category": "function",
    "text": "violates_KKT_condition(i2, distances_to_decision_boundary, α, C, opt_precision)\n\n\n\n\n\n"
},

{
    "location": "smo/#SVDD.smo",
    "page": "SMO",
    "title": "SVDD.smo",
    "category": "function",
    "text": "smo(α, K, C, opt_precision, max_iterations)\n\n\n\n\n\n"
},

{
    "location": "smo/#SVDD.examine_and_update_predictions!",
    "page": "SMO",
    "title": "SVDD.examine_and_update_predictions!",
    "category": "function",
    "text": "examine_and_update_predictions!(α, distances_to_center, distances_to_decision_boundary, R,\n    KKT_violations, black_list, K, C, opt_precision)\n\n\n\n\n\n"
},

{
    "location": "smo/#Selection-of-i2-1",
    "page": "SMO",
    "title": "Selection of i2",
    "category": "section",
    "text": "A minimum of P has to obey the KKT conditions. The relevant KKT condition here is complementary slackness, i.e.,  mu_i g_i(x^*) = 0  forall iwith dual variable mu and inequality conditions g. In other words, either the inequality constraint is fulfilled with equality, i.e., g_i = 0, or the Lagrange multiplier is zero, i.e., mu_i=0. For SVDD, this translates to  beginaligned\n    leftlVert a - phi(x_i) rightrVert^2  R^2 rightarrow alpha_i = 0 \n    leftlVert a - phi(x_i) rightrVert^2 = R^2 rightarrow  0  alpha_i  C\n    leftlVert a - phi(x_i) rightrVert^2  R^2 rightarrow alpha_i = C\n endalignedSee [2] for details. The distance to the decision boundary is leftlVert a - phi(x_i) rightrVert^2 - R^2 which is negative for observations that lie in the hypershpere.So to check for KKT violations, one has to calculate the distance of phi(x_i) from the decision boundary, i.e., the left-hand side of the implications above, and compare it with the the respective alpha value. The check for KKT violations is implemented in  SVDD.violates_KKT_conditionSVDD.smo selects i2 by searching for indices that violate the KKT conditions.  SVDD.smoThis function conducts two tyes of search.First type: search over the full data set, and randomly selects one of the violating indices.Second type: restricted search for violations over the subset where 0 alpha_i  C. These variables are the non-bounded support vectors SV_nb.There is one search of the first type, then multiple searches of the second type. After each search, i2 is selected randomly from one of the violating indices, seeSVDD.examine_and_update_predictions!"
},

{
    "location": "smo/#SVDD.second_choice_heuristic",
    "page": "SMO",
    "title": "SVDD.second_choice_heuristic",
    "category": "function",
    "text": "second_choice_heuristic(i2, α, distances_to_center, C, opt_precision)\n\n\n\n\n\n"
},

{
    "location": "smo/#SVDD.examineExample!",
    "page": "SMO",
    "title": "SVDD.examineExample!",
    "category": "function",
    "text": "examineExample!(α, i2, distances_to_center, K, C, opt_precision)\n\nThe fallback strategies if second choice heuristic returns false follow recommendations in\nJ. Platt, \"Sequential minimal optimization: A fast algorithm for training support vector machines,\" 1998.\n\n\n\n\n\n"
},

{
    "location": "smo/#Selection-of-i1-1",
    "page": "SMO",
    "title": "Selection of i1",
    "category": "section",
    "text": "SMO selects i1 such that the optimization step is as large as possible. The idea for selecting i1 is as follows. For alpha_i2  0 and negative distance to decision boundary, alpha may decrease. So a good alpha_i1 is one that is likely to increase in the optimization step, i.e., an index where the distance to the decision boundary is positive, and alpha_i1 = 0. The heuristic SMO selects the i1 with maximum absolute distance between the distance to the center of i2 and the distance to the center of some i1 in SV_nb. (Note that using the distance to the decision boundary is equivalent to using the distance to the center in this step). This selection heuristic is implemented in  SVDD.second_choice_heuristicIn some cases, the selected i1 does not lead to a positive optimization step. In this case, there are two fallback strategies. First, all other indices in SV_nb are selected, in random order, whether they result in a positive optimization step. Second, if there still is no i1 that results in a positive optimization step, all remaining indices are selected. If none of the fallback strategies works, i2 is skipped and added to a blacklist. The fallback strategies are implemented in  SVDD.examineExample!"
},

{
    "location": "smo/#Termination-1",
    "page": "SMO",
    "title": "Termination",
    "category": "section",
    "text": "If there are no more KKT violations, the algorithm terminates."
},

{
    "location": "smo/#Further-implementation-details-1",
    "page": "SMO",
    "title": "Further implementation details",
    "category": "section",
    "text": "This section describes some further implementation details."
},

{
    "location": "smo/#SVDD.initialize_alpha",
    "page": "SMO",
    "title": "SVDD.initialize_alpha",
    "category": "function",
    "text": "initialize_alpha(data, C)\n\n\n\n\n\n"
},

{
    "location": "smo/#Initialize-alpha-1",
    "page": "SMO",
    "title": "Initialize alpha",
    "category": "section",
    "text": "The vector alpha must be initialized such that it fulfills the constraints of D. The implementation uses the initialization strategy proposed in [3], i.e., randomly setting frac1C indices to C. This is implemented in  SVDD.initialize_alpha"
},

{
    "location": "smo/#SVDD.calculate_predictions",
    "page": "SMO",
    "title": "SVDD.calculate_predictions",
    "category": "function",
    "text": "calculate_predictions(α, K, C, opt_precision)\n\n\n\n\n\n"
},

{
    "location": "smo/#Calculating-Distances-to-Decision-Boundary-1",
    "page": "SMO",
    "title": "Calculating Distances to Decision Boundary",
    "category": "section",
    "text": "The distances to the decision boundary are calculated in  SVDD.calculate_predictionsIn general, to calculate R, one can calculate the distance to any non-bounded support vector, i.e., 0  alpha_i  C, as they all lie on the hypershpere. However, this may not always hold. There may be cases where the solution for R is not unique, and different support vectors result in different R, in particular in intermediate optimization steps where some alpha values may be non-bounded but violate the KKT conditions. Therefore, R is averaged over all non-bounded support vectors. See also [4] for details on non-unique R values."
},

{
    "location": "smo/#SMO-parameters-1",
    "page": "SMO",
    "title": "SMO parameters",
    "category": "section",
    "text": "There are two parameters for SMO: opt_precision and max_iterations.opt_precision influences the convergence. Small opt_precision values require a larger number of iterations until termination.max_iterations controls the number of times a new i2 is selected to attempt an optimization step."
},

{
    "location": "smo/#SVDD.solve!-Tuple{VanillaSVDD,SMOSolver}",
    "page": "SMO",
    "title": "SVDD.solve!",
    "category": "method",
    "text": "solve!(model::VanillaSVDD, solver::SMOSolver)\n\n\n\n\n\n"
},

{
    "location": "smo/#External-API-1",
    "page": "SMO",
    "title": "External API",
    "category": "section",
    "text": "  SVDD.solve!(model::VanillaSVDD, solver::SMOSolver)"
},

{
    "location": "smo/#References-1",
    "page": "SMO",
    "title": "References",
    "category": "section",
    "text": "[1] J. Platt, \"Sequential minimal optimization: A fast algorithm for training support vector machines,\" 1998.[2] D. M. J. Tax and R. P. W. Duin, \"Support Vector Data Description,\"\" Mach. Learn., 2004.[3] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, \"Estimating the support of a high-dimensional distribution,\"\" Neural Comput., 2001.[4] W.-C. Chang, C.-P. Lee, and C.-J. Lin, \"A revisit to support vector data description,\"Nat. Taiwan Univ., Tech. Rep, 2013."
},

]}
