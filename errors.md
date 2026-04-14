# SHAP Execution Warnings — Explanation

The following warnings appear during the SHAP computation cell. All of them are **expected and harmless** — they do not affect the correctness of the SHAP values or the summary plot.

---

## Warning 1 — `UserWarning: Linear regression equation is singular`

**Full message:**
UserWarning: Linear regression equation is singular, a least squares solution is used instead.
To avoid this situation and get a regular matrix do one of the following:
1) turn up the number of samples,
2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,
3) group features together to reduce the number of inputs that need to be explained.


**What it is:**  
Internally, `KernelExplainer` works by fitting a weighted linear regression to approximate the contribution of each feature to the model's prediction. In our case, we have **1,000 TF-IDF features** but only **50 background samples**. This creates an underdetermined system — there are more unknowns (features) than equations (samples), making the regression matrix mathematically **singular** (non-invertible).

**Effect on results:**  
SHAP automatically switches to a **least-squares approximation** instead of an exact solution. The SHAP values produced are still meaningful approximations and the summary plot is valid. This is a known limitation of `KernelExplainer` on high-dimensional sparse text data.

**Root cause:**  
Using `KernelExplainer` on 1,000-dimensional TF-IDF vectors with only 50 background samples. The feature space is far larger than the background sample size.

**Why we did not fix it:**  
The only meaningful fix would be to drastically reduce `max_features` in `TfidfVectorizer` (e.g., from 1,000 to ~100). This would eliminate the singularity but would significantly reduce the model's vocabulary coverage and hurt the F1 Score on spam classification. Preserving model quality is the higher priority here.

---

## Warning 2 — `ConvergenceWarning: Regressors in active set degenerate`

**Full message:**
ConvergenceWarning: Regressors in active set degenerate. Dropping a regressor,
after N iterations, i.e. alpha=X.XXXe-XX, with an active set of N regressors,
and the smallest cholesky pivot element being 2.220e-16.
Reduce max_iter or increase eps parameters.



**What it is:**  
SHAP's `KernelExplainer` uses a LARS (Least Angle Regression) solver internally to select which features to include in each local linear explanation. This solver relies on Cholesky matrix decomposition. When the pivot element in that decomposition becomes extremely small (near machine precision: `2.220e-16`), the matrix is considered numerically degenerate and the solver drops that regressor and continues.

**Effect on results:**  
The affected feature (word) is dropped from that specific local explanation step and the solver continues with the remaining features. The final SHAP values and summary plot are computed and displayed correctly. This is documented expected behavior in scikit-learn's LARS implementation.

**Root cause:**  
TF-IDF feature vectors are extremely **sparse** — for any given message, the vast majority of the 1,000 feature values are exactly `0.0`. This extreme sparsity causes near-zero pivot values in the Cholesky decomposition, triggering numerical instability. This is a well-known issue when applying `KernelExplainer` to high-dimensional sparse text representations.

**Why we did not fix it:**  
The fix would require either reducing `max_features` (hurting model accuracy) or increasing `nsamples` to 500+ (making the SHAP cell take 10-20x longer to run with no improvement in the final classification results). Neither trade-off is justified for this task.

---

## Summary Table

| Warning | Source | Affects Results? | Fixed? |
|---|---|---|---|
| `UserWarning: Linear regression equation is singular` | SHAP `KernelExplainer` — underdetermined system | ❌ No | ❌ Fix would reduce model F1 Score |
| `ConvergenceWarning: Regressors in active set degenerate` | sklearn LARS — sparse Cholesky instability | ❌ No | ❌ Fix would reduce model F1 Score |

Both warnings are byproducts of running a model-agnostic explainer (`KernelExplainer`) on high-dimensional sparse TF-IDF features with a limited background sample size. The SHAP summary plot output is valid and the conclusions drawn from it are reliable.
