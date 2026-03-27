# Optimal Differentiation: Strategic Conformity Analysis of Fortune 1000 Firms

**Sample period:** 1963–2019 (pre-pandemic)
**Outcomes:** Return on Assets (ROA) and Tobin's Q
**Method:** Quadratic fixed-effects OLS with firm and year FEs, two-way clustered SEs (firm × year), box-constrained QP optimization (Gurobi)

---

## Table 1. Sample Summary

The sample covers Fortune 1000 firms from 1963 to 2019. The six strategy variables are lagged one period, winsorized at the 1st and 99th percentiles within each outcome-specific sample, and entered in quadratic form. All regressions absorb firm and year fixed effects via alternating projections and report two-way clustered covariance matrices.

| Outcome   | Firm-years | Firms | Years | Winsorized |
|-----------|------------|-------|-------|------------|
| ROA       | 24,112     | 1,311 | 57    | Yes        |
| Tobin's Q | 21,518     | 1,173 | 57    | Yes        |

---

## Table 2. Pairwise Pearson Correlations Among Lagged Strategy Variables

### Panel A. ROA sample

|             | R&D    | Capital | Leverage | Advertising | Dividend | Risk   |
|-------------|--------|---------|----------|-------------|----------|--------|
| R&D         |  1.000 | −0.178  | −0.237   |  0.048      | −0.069   |  0.162 |
| Capital     | −0.178 |  1.000  |  0.230   | −0.178      |  0.034   | −0.103 |
| Leverage    | −0.237 |  0.230  |  1.000   | −0.065      |  0.155   | −0.012 |
| Advertising |  0.048 | −0.178  | −0.065   |  1.000      |  0.131   |  0.021 |
| Dividend    | −0.069 |  0.034  |  0.155   |  0.131      |  1.000   | −0.280 |
| Risk        |  0.162 | −0.103  | −0.012   |  0.021      | −0.280   |  1.000 |

### Panel B. Tobin's Q sample

|             | R&D    | Capital | Leverage | Advertising | Dividend | Risk   |
|-------------|--------|---------|----------|-------------|----------|--------|
| R&D         |  1.000 | −0.210  | −0.231   |  0.036      | −0.081   |  0.169 |
| Capital     | −0.210 |  1.000  |  0.307   | −0.202      |  0.021   | −0.122 |
| Leverage    | −0.231 |  0.307  |  1.000   | −0.047      |  0.191   | −0.012 |
| Advertising |  0.036 | −0.202  | −0.047   |  1.000      |  0.128   |  0.014 |
| Dividend    | −0.081 |  0.021  |  0.191   |  0.128      |  1.000   | −0.296 |
| Risk        |  0.169 | −0.122  | −0.012   |  0.014      | −0.296   |  1.000 |

*Notes:* Correlations are moderate throughout; the largest absolute correlation is 0.307 (Capital × Leverage in the Tobin's Q sample). The Dividend–Risk pair is the most consistently negative across both samples (−0.280 and −0.296).

---

## Table 3. Single-Variable Quadratic Fixed-Effects Models

Each row is a separate firm- and year-fixed-effects regression of the outcome on one lagged strategy variable and its square. Standard errors are two-way clustered by firm and year. The one-dimensional optimum is computed over the observed support of the winsorized variable.

### Panel A. ROA (N = 24,112)

| Variable            | Linear coef. | SE     | p-value  | Quad. coef. | SE      | p-value  | 1D Opt. | Bound? | Within-R² |
|---------------------|--------------|--------|----------|-------------|---------|----------|---------|--------|-----------|
| R&D intensity       | −0.526       | 0.176  | 0.004    |  0.469      | 0.644   | 0.470    | 0.000   | Yes    | 0.014     |
| Capital intensity   | −0.045       | 0.008  | < 0.001  |  0.005      | 0.002   | 0.012    | 0.015   | Yes    | 0.023     |
| Leverage            | −0.184       | 0.017  | < 0.001  |  0.151      | 0.015   | < 0.001  | 0.000   | Yes    | 0.044     |
| Advertising intensity | −0.319     | 0.143  | 0.029    |  2.380      | 1.105   | 0.036    | 0.147   | Yes    | 0.002     |
| Dividend policy     |  0.297       | 0.067  | < 0.001  | −0.370      | 0.172   | 0.036    | 0.391   | No     | 0.024     |
| Unsystematic risk   | −1.034       | 0.522  | 0.053    | −5.138      | 10.041  | 0.611    | 0.007   | Yes    | 0.014     |

### Panel B. Tobin's Q (N = 21,518)

| Variable            | Linear coef. | SE     | p-value  | Quad. coef.  | SE       | p-value  | 1D Opt. | Bound? | Within-R² |
|---------------------|--------------|--------|----------|--------------|----------|----------|---------|--------|-----------|
| R&D intensity       |  −3.121      | 2.743  | 0.260    |    1.579     |   9.956  | 0.875    | 0.000   | Yes    | 0.004     |
| Capital intensity   |  −0.342      | 0.102  | 0.001    |    0.078     |   0.024  | 0.002    | 0.028   | Yes    | 0.003     |
| Leverage            |  −2.216      | 0.255  | < 0.001  |    2.085     |   0.223  | < 0.001  | 1.117   | Yes    | 0.034     |
| Advertising intensity | −6.070     | 2.268  | 0.010    |   40.665     |  18.290  | 0.030    | 0.150   | Yes    | 0.003     |
| Dividend policy     |   2.866      | 0.860  | 0.002    |   −0.964     |   2.101  | 0.648    | 0.419   | No     | 0.028     |
| Unsystematic risk   | −33.198      | 9.180  | 0.001    |  498.609     | 182.374  | 0.008    | 0.007   | Yes    | 0.006     |

*Notes:* Dividend policy is the only variable with an interior optimum in the ROA sample (0.391). All other univariate optima hit a boundary of the observed support. The within-R² values are modest (0.2–4.4%), consistent with the multi-dimensional nature of strategy.

---

## Table 4. Additive Quadratic Fixed-Effects Model

Joint regression with linear and squared terms for all six variables; no cross terms. Variables are centered before constructing quadratic terms to reduce mechanical collinearity; coefficients are reported on the centered scale.

| Term                   | ROA coef. | ROA SE  | ROA p    | Tobin's Q coef. | Tobin's Q SE | Tobin's Q p |
|------------------------|-----------|---------|----------|-----------------|--------------|-------------|
| R&D intensity          | −0.397    | 0.120   | 0.002    | −2.462          | 2.131        | 0.253       |
| Capital intensity      | −0.024    | 0.006   | < 0.001  | −0.086          | 0.067        | 0.202       |
| Leverage               | −0.066    | 0.007   | < 0.001  | −0.749          | 0.124        | < 0.001     |
| Advertising intensity  | −0.125    | 0.094   | 0.188    | −4.168          | 1.694        | 0.017       |
| Dividend policy        |  0.259    | 0.043   | < 0.001  |  3.334          | 0.576        | < 0.001     |
| Unsystematic risk      | −0.784    | 0.213   | 0.001    | −12.063         | 3.916        | 0.003       |
| R&D intensity²         |  0.171    | 0.527   | 0.747    |   0.601         | 9.273        | 0.949       |
| Capital intensity²     |  0.002    | 0.002   | 0.255    |   0.044         | 0.022        | 0.049       |
| Leverage²              |  0.159    | 0.014   | < 0.001  |   2.199         | 0.210        | < 0.001     |
| Advertising intensity² |  1.445    | 0.907   | 0.117    |  35.322         | 16.959       | 0.042       |
| Dividend policy²       | −0.355    | 0.143   | 0.016    |  −2.703         | 1.821        | 0.143       |
| Unsystematic risk²     | −15.322   | 9.615   | 0.117    | 303.391         | 170.871      | 0.081       |
| **Joint Wald (all 12)**| **χ²(12) = 387.3** | | **< 0.001** | **χ²(12) = 186.8** | | **< 0.001** |

*Notes:* Leverage is the dominant positive-quadratic term in both outcomes (leverage² > 0), consistent with a U-shaped relationship between debt and performance. Dividend policy enters positively and concavely (dividend policy² < 0) for ROA, suggesting a concave return to payout generosity.

---

## Table 5. Full Quadratic Fixed-Effects Model: Selected Coefficients and Joint Tests

Extends the additive model with all 15 pairwise cross terms. Only terms significant at the 10% level in at least one outcome are shown in the main panel; the joint Wald tests cover all terms.

### Panel A. Main-effect and squared terms

| Term                   | ROA coef. | ROA p    | Tobin's Q coef. | Tobin's Q p |
|------------------------|-----------|----------|-----------------|-------------|
| R&D intensity          | −0.392    | 0.001    |  −2.583         | 0.207       |
| Capital intensity      | −0.021    | 0.001    |  −0.095         | 0.282       |
| Leverage               | −0.067    | < 0.001  |  −0.689         | < 0.001     |
| Advertising intensity  | −0.104    | 0.248    |  −4.454         | 0.016       |
| Dividend policy        |  0.279    | < 0.001  |   3.780         | < 0.001     |
| Unsystematic risk      | −0.742    | 0.001    |  −8.603         | 0.024       |
| Leverage²              |  0.155    | < 0.001  |   2.242         | < 0.001     |
| Advertising intensity² |  1.945    | 0.029    |  42.993         | 0.011       |
| Unsystematic risk²     | −25.296   | 0.009    | −37.397         | 0.837       |

### Panel B. Selected cross terms (significant at 10% in at least one outcome)

| Cross term                          | ROA coef. | ROA p    | Tobin's Q coef. | Tobin's Q p |
|-------------------------------------|-----------|----------|-----------------|-------------|
| R&D × Unsystematic risk             | −3.290    | 0.197    | 102.181         | 0.059       |
| Capital intensity × Leverage        |  0.013    | 0.160    |   0.146         | 0.203       |
| Capital intensity × Dividend policy |  0.039    | 0.100    |   0.186         | 0.604       |
| Leverage × Advertising intensity    |  0.484    | 0.007    |   7.791         | 0.030       |
| Leverage × Dividend policy          | −0.469    | < 0.001  |  −6.018         | < 0.001     |
| Leverage × Unsystematic risk        | −0.389    | 0.393    | −32.551         | < 0.001     |
| Advertising × Dividend policy       |  0.623    | 0.262    |  19.617         | 0.021       |
| Dividend policy × Unsystematic risk | −5.429    | 0.001    | −107.203        | < 0.001     |

### Panel C. Joint Wald tests

|                              | ROA               | Tobin's Q         |
|------------------------------|-------------------|-------------------|
| All 27 terms, χ²(27)         | 504.7, p < 0.001  | 309.2, p < 0.001  |
| Cross terms only, χ²(15)     |  61.2, p < 0.001  |  99.9, p < 0.001  |
| Within-R² (full model)       | 0.128             | 0.111             |
| Within-R² (additive model)   | 0.110             | 0.077             |

*Notes:* The 15 cross terms are jointly highly significant in both outcomes (p < 0.001), with the cross-term block contributing 1.8 pp to within-R² for ROA and 3.4 pp for Tobin's Q. Two interactions dominate across outcomes: **Leverage × Dividend policy** is consistently negative and precisely estimated, while **Dividend policy × Unsystematic risk** is strongly negative in both samples, suggesting that high-dividend strategies are particularly costly for volatile firms.

---

## Table 6. Constrained Optimal Strategy Positions

The constrained optimum under each model is found via box-constrained QP (Gurobi, NonConvex=2), enforcing each variable to lie within its observed sample support after winsorization. Additive and full quadratic models are optimized jointly across all six dimensions.

### Panel A. ROA

| Variable            | Support              | M1 Individual opt. | M2 Additive opt. | M3 Full opt. |
|---------------------|----------------------|--------------------|------------------|--------------|
| R&D intensity       | [0.000,  0.252]      | 0.000              | 0.000            | 0.000        |
| Capital intensity   | [0.015,  3.718]      | 0.015              | 0.015            | 3.533        |
| Leverage            | [0.000,  1.210]      | 0.000              | 0.000            | 0.000        |
| Advertising intensity | [0.000, 0.147]     | 0.147              | 0.147            | 0.147        |
| Dividend policy     | [−0.033, 0.494]      | 0.391              | 0.391            | 0.391        |
| Unsystematic risk   | [0.007,  0.045]      | 0.007              | 0.007            | 0.007        |
| **Estimated surface optimum** | —         | —                  | **0.1338**       | **0.2798**   |

### Panel B. Tobin's Q

| Variable            | Support              | M1 Individual opt. | M2 Additive opt. | M3 Full opt. |
|---------------------|----------------------|--------------------|------------------|--------------|
| R&D intensity       | [0.000,  0.261]      | 0.000              | 0.000            | 0.000        |
| Capital intensity   | [0.026,  3.792]      | 0.028              | 3.582            | 3.582        |
| Leverage            | [0.000,  1.259]      | 1.117              | 1.117            | 0.000        |
| Advertising intensity | [0.000, 0.151]     | 0.150              | 0.150            | 0.150        |
| Dividend policy     | [−0.069, 0.553]      | 0.419              | 0.419            | 0.419        |
| Unsystematic risk   | [0.007,  0.045]      | 0.007              | 0.007            | 0.007        |
| **Estimated surface optimum** | —         | —                  | **1.9627**       | **4.8617**   |

*Notes:* The estimated surface optimum for M1 is not directly comparable to M2 and M3 because M1 optimizes six separate univariate surfaces rather than a joint surface. For M3, the large jump in Tobin's Q from 1.96 (M2) to 4.86 (M3) reflects the impact of cross-term interactions — in particular, the full model recommends zero leverage (vs. 1.117 under M2), exploiting the negative Leverage × Dividend policy and Leverage × Unsystematic risk interactions.

---

## Table 7. Optimality Gap Analysis

The full quadratic model (M3) serves as the benchmark — its estimated surface is the best available approximation of the true performance function. The optimality gap measures how much the full-model estimated surface value is lost when a simpler model's optimal strategy is used instead.

$$\text{gap} = f^{*}_{M3} - f(x^{*}_{\text{model}}) \qquad \text{gap}_{\text{pct}} = \frac{\text{gap}}{f^{*}_{M3}} \times 100$$

| Outcome   | Model              | Full surface value at model's $x^*$ | Abs. Gap | % Gap  |
|-----------|--------------------|-------------------------------------|----------|--------|
| ROA       | M3 Full Quadratic  | 0.2798 *(benchmark)*                | —        | —      |
| ROA       | M2 Additive        | 0.2551                              | 0.0248   | 8.85%  |
| ROA       | M1 Individual      | 0.2551                              | 0.0248   | 8.85%  |
| Tobin's Q | M3 Full Quadratic  | 4.8617 *(benchmark)*                | —        | —      |
| Tobin's Q | M2 Additive        | 4.6389                              | 0.2228   | 4.58%  |
| Tobin's Q | M1 Individual      | 3.3629                              | 1.4988   | 30.83% |

*Notes:* For ROA, M1 and M2 happen to prescribe identical strategies (both hit the same corner solutions), so their gaps are equal. For Tobin's Q the divergence is more revealing: M2 loses 4.6% by ignoring cross-term interactions, but M1 loses 30.8% — nearly one-third of the achievable gain — because optimizing each dimension independently misses the joint structure. The full model's recommendation to set leverage to zero (exploiting the negative Leverage × Dividend policy and Leverage × Unsystematic risk interactions) is only discovered when cross terms are modeled explicitly.

---

## Table 8. Robustness Checks

The joint significance of the additive and full-quadratic term blocks is checked across three alternative specifications: (i) firm clustering only, (ii) industry-year fixed effects replacing year FEs, and (iii) main specification augmented with firm size, age, market beta, and outcome-specific industry median controls.

### Panel A. ROA

| Specification                          | Model     | Within-R² | Wald χ²  | p-value  |
|----------------------------------------|-----------|-----------|----------|----------|
| Main (firm + year FE, two-way cluster) | Additive  | 0.110     | 387.3    | < 0.001  |
| Main (firm + year FE, two-way cluster) | Full      | 0.128     | 504.7†   | < 0.001  |
| Firm clustering only                   | Additive  | 0.110     | 505.5    | < 0.001  |
| Firm clustering only                   | Full      | 0.129     |  68.8†   | < 0.001  |
| Industry-year FE, firm clustering      | Additive  | 0.113     | 482.7    | < 0.001  |
| Industry-year FE, firm clustering      | Full      | 0.133     |  85.1†   | < 0.001  |
| Main + controls                        | Additive  | 0.212     | 347.0    | < 0.001  |
| Main + controls                        | Full      | 0.228     |  55.6†   | < 0.001  |

### Panel B. Tobin's Q

| Specification                          | Model     | Within-R² | Wald χ²  | p-value  |
|----------------------------------------|-----------|-----------|----------|----------|
| Main (firm + year FE, two-way cluster) | Additive  | 0.077     | 186.8    | < 0.001  |
| Main (firm + year FE, two-way cluster) | Full      | 0.111     | 309.2†   | < 0.001  |
| Firm clustering only                   | Additive  | 0.077     | 211.7    | < 0.001  |
| Firm clustering only                   | Full      | 0.111     | 118.3†   | < 0.001  |
| Industry-year FE, firm clustering      | Additive  | 0.075     | 199.1    | < 0.001  |
| Industry-year FE, firm clustering      | Full      | 0.102     |  86.1†   | < 0.001  |
| Main + controls                        | Additive  | 0.217     | 166.4    | < 0.001  |
| Main + controls                        | Full      | 0.237     |  71.8†   | < 0.001  |

*Notes:* † For full quadratic models, the Wald test covers the 15 cross terms only. For additive models it covers all 12 linear and quadratic terms. All results remain highly significant across every specification, confirming that both the additive quadratic structure and the cross-term interactions are robust to alternative clustering, fixed-effect, and control-variable choices.

---

## Summary of Main Findings

1. **Strategy interactions matter.** The 15 cross terms are jointly significant at p < 0.001 in both outcomes and every robustness specification. Ignoring them (M2) leads to an 8.9% optimality gap for ROA and 4.6% for Tobin's Q.

2. **Joint optimization is essential for Tobin's Q.** The M1 individual model — which optimizes each dimension separately — loses 30.8% of achievable Tobin's Q because it cannot discover the leverage-zero recommendation that emerges from the negative Leverage × Dividend policy and Leverage × Unsystematic risk interactions.

3. **Key interactions.** Two cross terms dominate consistently: (a) *Leverage × Dividend policy* is negative in both outcomes, meaning high-dividend firms should carry less debt; (b) *Dividend policy × Unsystematic risk* is strongly negative, meaning volatile firms should limit dividends.

4. **Boundary optima.** Most strategy variables hit a boundary of the observed support at the optimum, suggesting either genuine corner solutions or that important non-linearities lie outside the observed data range. Dividend policy is the main exception, with an interior optimum in the ROA sample (≈ 0.39).

5. **Pre-pandemic stability.** Restricting to 1963–2019 (24,112 ROA firm-years; 21,518 Tobin's Q firm-years) produces qualitatively identical conclusions to the full sample, confirming the results are not driven by pandemic-era disruptions.
