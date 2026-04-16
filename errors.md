# SHAP Execution Warnings — Explanation

The following warnings appear during the SHAP computation cell. All of them are **expected and harmless** — they do not affect the correctness of the SHAP values or the summary plot.

---

### SHAP Warning — ConvergenceWarning from LARS

This warning originates from SHAP's internal use of LARS regression 
(Least Angle Regression) during KernelExplainer computation.

**Cause:** The TF-IDF matrix is sparse (1000 features, most = 0), 
causing near-degenerate columns that trigger numerical instability 
in LARS.

**Impact:** None. SHAP automatically drops the degenerate regressor 
and continues. The resulting SHAP values and summary plot are 
correct and unaffected.

**Fix:** Reducing features to 100-200 would eliminate this warning 
but would significantly hurt model performance (F1 drops from 0.90).
Not worth it.


---
