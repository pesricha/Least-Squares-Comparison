# Least Squares Polynomial Approximation

This project implements a closed-form solution to the least squares approximation problem, applied to the function \( f(t) = \sin(10t) \) over the interval \( t \in [0, 1] \). We approximate this function with a 14th-degree polynomial using 100 data points and various matrix factorization methods. The primary goal is to compare the accuracy and stability of different algorithms for solving the least squares polynomial coefficients.

## When to Use Closed-Form vs. Iterative Solutions

- **Closed-Form Solutions**: Effective when the design matrix \( A \) is of moderate size and well-conditioned, allowing efficient direct computation with factorization techniques.
- **Iterative Solutions** (e.g., Gradient Descent): Preferable for very large systems where matrix factorization is computationally prohibitive, or when data is sparse and iterative refinement provides a faster path to convergence without requiring full decomposition of \( A \).


## Problem Description

Given data points \((t_i, f(t_i))\), where \( f(t) = \sin(10t) \), the goal is to fit a polynomial \( P(t) \) of degree 14 to the data by solving the linear system:
\[
A c = f
\]
where \( A \) is the design matrix, \( c \) is the vector of polynomial coefficients, and \( f \) is the target function evaluated at each \( t_i \).

## Algorithms Compared

1. **Modified Gram-Schmidt (MGS)**  
   Solves \( A c = f \) using QR Factorization via Modified Gram-Schmidt. Here, \( A = QR \) where \( Q \) is orthogonal, and \( R \) is upper triangular. The back-substitution is implemented for solving the system \( Rc = Q^T f \).

2. **Householder Transformation (HHT)**  
   Uses Householder reflections to obtain an orthogonal factor \( Q \) and an upper triangular \( R \) such that \( A = QR \). The system is then solved by back-substitution.

3. **Singular Value Decomposition (SVD)**  
   Using SVD, \( A \) is decomposed as \( A = U \Sigma V^T \). The least squares solution is obtained by solving the system with \( \Sigma \) and \( V \).

4. **Normal Equations**  
   Using normal equations, we solve \( A^T A c = A^T f \), which is computed directly with MATLAB’s backslash operator (\), accepted here as the "true" least squares solution.

## Error Analysis

We evaluate the least squares error for each method:
\[
\text{Least Squares Error} = \| f_{\text{true}} - f_{\text{pred}} \|_2^2
\]
The observed errors and stability properties are summarized below:

- **MGS**: Moderate accuracy; prone to instability due to loss of orthogonality in \( Q \).
- **Householder**: High accuracy; stable due to robust orthogonalization.
- **SVD**: High accuracy and stability, though computationally intensive.
- **Normal Equations**: Lowest accuracy; unstable for ill-conditioned problems due to squaring of the condition number \( \kappa(A)^2 \).

## Results Summary

The stability and accuracy of each method were measured by comparing the predicted coefficients and relative errors. The SVD and Householder methods yielded the best approximations, with stability preserved by their orthogonalization properties. Normal Equations showed significant instability due to amplified rounding errors in \( A^T A \).

| Method                | Least Squares Error            |
|-----------------------|--------------------------------|
| Modified Gram-Schmidt | \(9.40 \times 10^{-14}\)       |
| Householder           | \(1.13 \times 10^{-15}\)       |
| SVD                   | \(1.13 \times 10^{-15}\)       |
| Normal Equations      | \(1.82 \times 10^{-7}\)        |

## Comparison of \( x^{14} \) Coefficients

For simplicity, we are going to compare the coefficient of \( x^{14} \) for each of the methods.

| Method                | Relative Error (Coefficient of \( x^{14} \)) |
|-----------------------|---------------------------------------------|
| Modified Gram-Schmidt | \(8.237193 \times 10^{-3}\)                 |
| Householder           | \(6.151479 \times 10^{-9}\)                 |
| SVD                   | \(2.406253 \times 10^{-8}\)                 |
| Normal Equations      | \(2.531678 \times 10^{0}\)                  |

**Table 1**: Relative error for \( x^{14} \) coefficient using different methods.

| Condition Component | \( y \) | \( x \) |
|---------------------|---------|---------|
| \( b \)            | \(1.000000 \times 10^{0}\)  | \(6.187516 \times 10^{3}\) |
| \( A \)            | \(3.016178 \times 10^{8}\)  | \(3.036732 \times 10^{8}\) |

**Table 2**: Condition numbers of \( y \) and \( x \) with respect to perturbations in \( b \) and \( A \).

## Explanation of Error Growth

At first glance, the observed relative error growth seems problematic, but this can be explained with the following arguments (with machine epsilon \( \epsilon_{m} \approx 2.22 \times 10^{-16} \)):

1. **Modified Gram-Schmidt (MGS)**:  
   The rounding error is on the order of \(10^{-3}\), indicating an amplification of \(10^{13}\), which greatly exceeds the condition number of the matrix. This error results from **instability** in the Modified Gram-Schmidt process, which often fails to produce a perfectly orthonormal \( Q \), adversely affecting the algorithm's accuracy. Therefore, MGS is **UNSTABLE**.

2. **Householder Transformation (HHT)**:  
   The rounding error here is around \(10^{-9}\), suggesting amplification by \(10^7\). Given that the condition number of \( x \) with respect to perturbations in \( A \) is approximately \(3 \times 10^8\), the inaccuracy in the \( x^{14} \) coefficient can be entirely attributed to **ill-conditioning**, not instability. Thus, HHT is **STABLE**.

3. **Singular Value Decomposition (SVD)**:  
   The SVD method yields similar results to Householder; the observed error can be attributed to the condition number of the system, indicating that SVD is also **STABLE**.

4. **Normal Equations**:  
   The error growth here is on the order of \(10^{16}\), explained by the squared condition number \( \kappa(A)^2 \approx 10^{16} \), which amplifies the error significantly. This behavior aligns with the theoretical result:
   \[
   \frac{\| \tilde{x} - x \|}{\| x \|} = O \left[( \kappa(A) + \kappa(A)^2 \frac{\tan\theta}{\eta} ) \epsilon_m\right]
   \]
   Depending on \( \theta \) and \( \eta \), \( \frac{\| \tilde{x} - x \|}{\| x \|} \) could be either \( \kappa(A)^2 \) or \( \kappa(A) \). Thus, Normal Equations are typically **UNSTABLE** for **ill-conditioned problems with close fits**.
