Option Normalisation using Historical Data

Premiums calculated for each period and vol quantile:

    C_{eu}(K|Q_{vol}) = E[e^-rT (R-K)+|Q_{vol}]
    P_{eu}(K|Q_{vol}) = E[e^-rT (K-R)+|Q_{vol}]

    where R = S_T/S_0, K = K/S_0

Each return has its own risk free rate, so separate discount applied to each point, instead of appluing single discount
to aggregated premium.

### Estimating Mean E[R]

Estimating from historicaly realised

    E_hist[R] = exp(E[log R] + 0.5*Scale[log R]^2)
    E_pred[R]  = model(period, vol | P)
    P ~ min L2[weight (log E_pred[R] - log E_hist[R])]

Positive scale used, to avoid inflating mean by negative skew, although effect is minimal.

Loss is weighted, to make errors equal across vols and periods.

1 and 9 vol deciles boost, as they look to be well shaped. As a side effect, the model underestimates mean at
long >730d periods, it's desired, because the dataset has survivorship bias.

Vol decile 10 ignored, it's too noisy. As result model understimates mean for 10 vol decile, it's desired.

Longer periods calculated with overlapping step 30d, and less reliably, so lowering weight a bit.

    weight = 1/period^2/vol^0.5
    weight[vol_dc in (1, 9)] *= 1.5

Found params: [-0.0011, 0.0075, 0.0427, 3.2455, 0.0037, -0.0043], loss: 1.1690

### Mean E[R | T, vol]

![Mean E[R], by period and vol (model - solid lines)](readme/mean-e-r-by-period-and-vol-model-solid-lines.png)

![Mean E[R]](readme/mean-e-r.png)

### Estimating Scale[log R]

Estimating from historicaly realised

    Scale_pred[log R] = model(period, vol | P)
    P ~ min L2[weight(Scale_pred[log R] - Scale_hist[log R])]

Loss is weighted, to make errors equal across vols and periods. Longer periods calculated with overlapping step 30d,
and less reliably, so lowering weight a bit.

    weight = 1/Scale_hist[log R]/period^0.5

Found params: [-0.5207, 1.9198, 1.2038, -0.1444, 0.2549, 0.0000, 0.0000, -0.0349, -0.0437], loss: 1.6550

### Scale[log R | T, vol]

![Estimated Scale (at expiration)](readme/estimated-scale-at-expiration.png)

![Vol by period, as EMA((log r)^2)^0.5](readme/vol-by-period-as-ema-log-r-2-0-5.png)

### Strike normalisation

    E_pred[R]         = predict_mmean(period, vol | P)
    Scale_pred[log R] = predict_scale(period, vol | P)
    E_pred[log R] = log E[R]_pred - 0.5*Scale_pred[log R]^2
    m = (log(K) - E_pred[log R])/Scale_pred[log R]

Compared to true normalised strike

    m_true = (log(K) - E_hist[log R])/Scale_hist[log R]

Normalising strike using mean, scale is biased as doesn't account for the distribution shape (skew, tails). But
should be consistent across periods and volatilities, as distribution should be similar.

Thre's minor mistake `E[log R] = log E[R] - 0.5*Scale[log R]^2` should use positive part of scale, but error is
very small, ignoring.

![Normalised Strikes vs True Normalised Strikes](readme/normalised-strikes-vs-true-normalised-strikes.png)

### Premium

Raw Strike K

![Premium P, Raw Strike K](readme/premium-p-raw-strike-k.png)

![Premium P, Raw Strike K, log scale](readme/premium-p-raw-strike-k-log-scale.png)

Norm Strike (log K - E[log R])/Scale[log R] (z score in log space or d2 from BlackScholes)

![Premium P, Norm Strike (log K - E[log R])/Scale[log R]](readme/premium-p-norm-strike-log-k-e-log-r-scale-log-r.png)

![Premium P, Norm Strike (log K - E[log R])/Scale[log R], log scale](readme/premium-p-norm-strike-log-k-e-log-r-scale-log-r-log-scale.png)

Norm Strike P(R < K | vol) (probability of ITM or F(d2) from BlackScholes)

![Premium, Norm Strike P(R < K | vol)](readme/premium-norm-strike-p-r-k-vol.png)

### Norm Premium

Normalising premium as P/E[R]/Scale[log R]

![Norm Premium P/E[R]/Scale[log R], Norm Strike (log K - E[log R])/Scale[log R]](readme/norm-premium-p-e-r-scale-log-r-norm-strike-log-k-e-log-r-scale-log-r.png)

![Norm Premium P/E[R]/Scale[log R], Norm Strike (log K - E[log R])/Scale[log R], log scale](readme/norm-premium-p-e-r-scale-log-r-norm-strike-log-k-e-log-r-scale-log-r-log-scale.png)

![Norm Premium P/E[R]/Scale[log R], Norm Strike P(R < K | vol)](readme/norm-premium-p-e-r-scale-log-r-norm-strike-p-r-k-vol.png)

![Ratio of Premium Min / Exp (calls solid)](readme/ratio-of-premium-min-exp-calls-solid.png)

#note bounds for american call: eu < am < 2eu

### Skew

![scalen_t2 vs scalep_t2, x - sort(period,vol)](readme/scalen-t2-vs-scalep-t2-x-sort-period-vol.png)

![MMean E[R] with scale vs scalep, x - sort(period,vol)](readme/mmean-e-r-with-scale-vs-scalep-x-sort-period-vol.png)

### Data

    - period - period, days

    - vol_dc - volatility decile 1..10
    - vol    - current moving daily volatility, as EWA(log(r)^2)^0.5, (scale unit, not variance), median of vol_dc group.

    - lmean_t2  - E[log R]
    - scale_t2  - Scale[log R] = mean_abs_dev(log R - lmean_t2) * sqrt(pi/2)

    - k  - strike
    - kq - strike quantile

    - p_exp - realised put premium using price at expiration (lower bound, european option)
    - p_max - realised put premium, using min price during option lifetime (upper bound, max possible
      for american option).
    - p_itm - realised probability of put ITM

    - c_exp - realised call premium using price at expiration (lower bound, european option)
    - c_max - realised call premium, using max price during option lifetime (upper bound, max possible
      for american option).
    - c_itm - realised probability of call ITM


Daily prices for 250 stocks all starting from 1973 till 2025, stats aggregated with moving window with step 30d, so
larger periods have overlapping. Dividends ignored. Data has survivorship bias, no bankrupts.

Data adjusted by adding bankrupts. The **annual bankruptsy probability** conditional on company volatility with total
annual rate P(b|T=365) = 0.5% to drop to x0.1.

