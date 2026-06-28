# Support / Resistance Zone Prediction

A machine-learning system that predicts **where a stock is likely to find support
and resistance** — both as a **premarket map** for the upcoming session and as a
**live intraday map** that updates as new bars arrive. Input is ordinary
multi-timeframe OHLCV candle data (≈ 1 month of history at 1m / 5m / 15m / 1h /
daily).

The central design decision: this is **not** a price-direction predictor. It is
**heatmap regression over an ATR-normalized price grid**. The model outputs, for
every price level around the current price, two intensities in `[0, 1]` —
*support intensity* and *resistance intensity* — and those heatmaps are
post-processed into a short ranked list of zones. Think object detection, but in
one dimension (price) instead of two (pixels).

---

## 1. Quickstart

```bash
pip install -r requirements.txt

# End-to-end demo on synthetic data (generates data, trains, evaluates,
# writes plots + sample maps to outputs/)
python scripts/run_pipeline.py --days 110                       # premarket only
python scripts/run_pipeline.py --days 95 --intraday --intraday-days 12

# Real data (identical pipeline; only the data source changes)
export POLYGON_API_KEY=...
python scripts/fetch_polygon.py --tickers AAPL MSFT NVDA --days 60 --intraday
```

Everything runs CPU-only in a few minutes. There is **no GPU dependency** — the
V1 model is gradient-boosted trees (LightGBM), which is the right tool for
tabular features at this data scale.

---

## 2. Why a heatmap, not a line or a label?

Real support/resistance is **fuzzy, multiple, and graded**: a stock has several
levels at once, each with a different strength, and each is a *zone* a few cents
or dollars wide rather than an exact price. Three naive framings all fail:

| Framing | Why it breaks |
|---|---|
| "Will price go up or down?" | Answers a different question. S/R is about *where reactions happen*, not net drift. |
| "Predict the single S/R price." | There are many levels; forcing one discards most of the structure. |
| "Classify each level as S or R." | You first need the candidate levels — that is the actual hard part. |

A **per-bin intensity heatmap** sidesteps all of this. It naturally represents
*multiple* levels, *graded* strength, and *zone width* (a bump, not a spike),
and it degrades gracefully (a vague level is a low broad bump, a strong level is
a tall sharp one). Discrete zones are recovered afterward by peak-picking.

---

## 3. Price representation — the key to transfer

Raw dollar prices are useless as model inputs: a `$3` move means nothing without
context, and a `$45` stock and a `$1600` stock live on totally different scales.
We represent everything as **signed distance from a causal reference price,
measured in ATR units, then binned**:

```
offset_in_atr = (price - reference_price) / ATR
bin_index     = round(offset_in_atr / 0.10) + center
```

* **Reference price** is causal: prior close for the premarket map, last trade
  for the intraday map.
* **ATR** (14-period, Wilder, computed only from bars ≤ t) is the volatility
  yardstick.
* Grid spans **±6 ATR** at **0.10 ATR per bin** → **121 bins**.

This makes targets and features **scale-free and transferable**: a level "1.5
ATR below price with two prior touches" looks identical whether the stock is
\$12 or \$1600. One model trains across the whole universe, and ATR-normalization
is what lets it generalize to tickers and volatility regimes it has not seen.

---

## 4. Labels — derived from what actually happened next

Labels come from **future price reactions**, never from hindsight lines drawn by
a human. Construction (see `src/labeling/`):

1. **Detect swings.** Fractal pivots: a swing high is a bar whose high is the max
   over a ±`swing_k`=3 window (swing low symmetric). Swing highs seed
   *resistance*; swing lows seed *support*.
2. **Score each swing's reaction strength.** For each pivot we measure the
   forward reversal magnitude in ATR (capped at 3 ATR), multiplied by how many
   times the level was subsequently *touched* (re-tested within
   `touch_tol_atr`=0.15 ATR) and weighted by volume. A level that was hit once
   and ignored scores low; a level that reversed price hard and was respected
   repeatedly scores high.
3. **Smear onto the grid.** Each scored level becomes a **Gaussian bump**
   (σ = 0.20 ATR) on the appropriate channel — this encodes the "zone, not line"
   intuition and gives the model a soft target.
4. **Normalize** each channel to `[0, 1]` per example (the strongest level in the
   window is 1.0).

The forward window is the *next session* for the premarket task and the *next 60
minutes* for the intraday task. Because targets are soft probabilities, the
model is trained with a **cross-entropy / logloss** objective per bin rather than
hard classification.

---

## 5. Leakage avoidance — taken seriously

Look-ahead bias is the single easiest way to build a model that looks brilliant
and is worthless. Controls baked into the pipeline:

* **Strict time split per example.** Features use only bars with timestamp ≤ t;
  labels use only the forward window (> t). The reference price and ATR are
  computed from data ≤ t.
* **Causal indicators.** ATR, EMAs, RSI, realized vol, session VWAP — all
  computed on trailing windows; no centered or forward-filled statistics.
* **The swing detector is honest about confirmation lag.** A ±k fractal can only
  be *confirmed* k bars later. Swings are used only for **label** construction
  (which lives strictly in the forward window); the *feature* side uses swing
  evidence only from already-closed bars, so a swing never informs a feature
  before it could have been known.
* **Embargoed walk-forward split by date.** Train / validation / test are
  split chronologically with a 2-day **embargo** between segments so a forward
  label window from the train set cannot overlap the start of validation. The
  embargo auto-shrinks (with a warning) on very short histories so early
  stopping still has a validation set, while never letting a date appear in two
  splits.
* **Per-example normalization** uses only that example's own grid — no global
  statistic computed across the test set leaks backward.

---

## 6. Features (`src/features/feature_builder.py`)

Each example is a **121-row frame** (one row per price bin) with **38 features**.
Two kinds:

**Per-bin (vary down the grid)** — the spatial signal the model localizes on:

* Signed and absolute ATR distance from the reference price.
* Round-number proximity (psychological levels).
* Volume-profile mass at this bin on 5m / 15m / 1h / daily (how much trading
  happened here historically).
* Touch counts at this bin per timeframe.
* Swing-high / swing-low evidence at this bin per timeframe, and confluence
  (agreement across timeframes).
* Distance from prior-day high / low / close, session open, and (intraday)
  session VWAP.

**Global (broadcast to every bin)** — the context that scales the whole map:

* ATR as a % of price, multi-horizon returns, daily RSI, trend slopes (daily and
  5m), realized volatility, where price sits in the day's range, overnight gap in
  ATR, time-of-day, minutes since open, day-of-week.

The top gain-importance features come out exactly as domain knowledge predicts:
`abs_dist_atr` (the base-rate prior that levels cluster near price),
`swing_lo_5m` driving **support**, `swing_hi_5m` driving **resistance**,
volume-profile mass, and `atr_pct`. The clean support↔resistance asymmetry in
the importances is direct evidence the model is learning real structure rather
than memorizing.

---

## 7. Premarket vs intraday — one engine, two configs

Both tasks use the *same* feature builder, grid, labeler, model, and metrics.
Only the conventions differ:

| | Premarket map | Intraday map |
|---|---|---|
| Reference price | Prior close | Last trade at time t |
| ATR source | Daily ATR | 5-minute ATR |
| Label window | Next session (5m bars) | Next 60 min (1m bars) |
| Example cadence | One per ticker per session | Every 30 min after a 15-min warmup |
| Output | One map for the day ahead | A map that refreshes each bar |

Live updating is handled by `IntradayServer` (`src/serving/`): feed it new 1m
bars as they close and it rebuilds higher timeframes, recomputes the causal
reference/ATR, and re-emits the ranked zones.

---

## 8. Evaluation — two complementary views

S/R quality has to be judged at two levels, because "did mass land on the right
bins" and "are the *zones* you'd actually trade correct" are different questions.

**(a) Per-bin probabilistic quality.** Average Precision (AP) and ROC-AUC of
predicted vs binarized true intensity. Rewards putting probability mass on the
right levels and correctly typing each as support vs resistance.

**(b) Zone-level usefulness (the metric that matters).** Both predicted and true
heatmaps are reduced to a small set of **discrete levels** via non-max-suppressed
peak picking. A predicted level *hits* if it lands within `hit_tol_atr`=0.25 ATR
of a true reaction level. Reported as **precision@K / recall@K** (K ∈ {3,5,8})
and **median localization error** in ATR. This is the common currency that lets a
learned heatmap and a pivot-point calculator be scored on exactly the same
footing — every method becomes "a ranked list of candidate price levels."

Zone metrics are reported **per channel** (support zones and resistance zones
separately) and on a polarity-blind **combined** channel. The per-channel view is
the one that matters for trading (is this a *bounce* zone or a *rejection*
zone?), and it is where a learned model's ability to *type* a level — which no
single rule-based baseline can do — shows up.

---

## 9. Baselines — the honesty check (`src/eval/baselines.py`)

A model is only interesting if it beats the rules a trader already uses. Every
classic S/R construction is implemented and scored with the identical zone
metric:

* **null_blob** — a single level at the current price. The crucial null
  hypothesis: "S/R is just wherever price is now." Beating this proves the model
  adds information beyond the `abs_dist_atr` base rate.
* **prev_day_hl** — prior session high & low.
* **pivots** — floor-trader pivots (P, R1, S1, R2, S2).
* **atr_bands** — reference ± 1 and ± 2 ATR.
* **round_numbers** — psychological round levels inside the grid.
* **opening_range** — opening-range high/low (intraday).
* **vwap_bands** — session VWAP ± 1 ATR (intraday).
* **volume_poc** — volume-profile POC and value-area edges.

---

## 10. Results (on synthetic data — read the caveat in §11)

Representative held-out (walk-forward test) numbers:

**Premarket**

* Per-bin: support AP ≈ 0.60, resistance AP ≈ 0.62, AUC ≈ 0.95, **≈ 7× lift**
  over base rate.
* Support zones F1@3: **model 0.52** vs prev-day-high/low 0.51 vs volume-POC 0.42
  vs null 0.36.
* Resistance zones F1@3: **model 0.53** vs prev-day 0.47 vs null 0.42.
* Lowest localization error of any method (≈ 0.06–0.07 ATR vs ≈ 0.10–0.11 for
  prev-day levels).
* On the polarity-blind combined metric the model **ties** prior-day high/low —
  exactly as expected, because prior-day extremes genuinely *are* strong S/R.

**Intraday** (where fusing many weak signals matters most)

* Per-bin: support AP ≈ 0.70, resistance AP ≈ 0.74, AUC ≈ 0.91–0.92, **≈ 7×
  lift**.
* Support zones F1@3: **model 0.65** vs VWAP-bands 0.34 vs null 0.31 — the model
  roughly **doubles** the best baseline.
* Resistance zones F1@3: **model 0.60** vs null 0.33 vs VWAP 0.31.

Interpretation: the model wins decisively where the use case actually needs it
(per-channel zones, and intraday especially), beats the critical null
("blob at current price") everywhere, and only ties — never loses to — the
single strongest classic rule on the deliberately polarity-blind aggregate.

---

## 11. Honest caveats and failure modes

* **These numbers are on synthetic data.** The generator (`src/data/synthetic.py`)
  injects realistic structure — persistent horizontal levels with mean-reverting
  reaction force, repeated touches, breakouts, round-number magnetism, U-shaped
  intraday volume — specifically so the *whole pipeline* can be validated
  end-to-end where the Polygon API is unreachable. But it is still a model of the
  market, not the market. The **modest premarket margin over prior-day levels is
  partly an artifact**: the synthetic levels are well-captured by prior-day
  extremes. On real data, where S/R arises from many noisier, overlapping sources
  (institutional volume nodes, options strikes, multi-timeframe confluence), the
  learned model's edge should *widen* — but that must be **re-measured on real
  Polygon data**, not assumed.
* **"Price visits levels anyway."** A level near current price is easy to "hit"
  simply because price wanders nearby. This is exactly why the `null_blob`
  baseline and the localization-error metric exist — to separate genuine skill
  from the base rate. Always read the model against the null, not in isolation.
* **Label noise.** Swing/reaction labels are a heuristic proxy for "true" S/R;
  there is no ground-truth oracle. Different swing-window or touch-tolerance
  settings shift the labels, so results should be checked for stability across
  label hyperparameters.
* **Regime dependence and overfitting.** A model trained on one volatility regime
  or a handful of tickers can overfit it. The universe should be broad and the
  walk-forward window long; per-ticker and per-regime breakdowns are advisable
  before trusting it live.
* **Not financial advice.** This is a research scaffold. Predicted zones are
  probabilistic and will be wrong regularly; position sizing and risk management
  are out of scope.

---

## 12. Repository layout

```
sr_system/
├── src/
│   ├── config.py                 # all hyperparameters (dataclasses)
│   ├── data/
│   │   ├── schema.py             # canonical tz-aware OHLCV bar schema
│   │   ├── synthetic.py          # realistic synthetic market generator
│   │   ├── polygon_client.py     # real Polygon REST client (stdlib only)
│   │   └── timeframes.py         # session-aware 1m -> 5m/15m/1h/1d resampling
│   ├── labeling/
│   │   ├── swings.py             # fractal swing detection
│   │   ├── reactions.py          # reaction-strength scoring
│   │   └── heatmap_labels.py     # premarket/intraday soft-label construction
│   ├── features/
│   │   ├── indicators.py         # causal ATR/EMA/RSI/VWAP/realized-vol
│   │   ├── price_grid.py         # ATR-normalized grid + Gaussian smearing
│   │   ├── volume_profile.py     # volume/touch profiles, POC/value-area
│   │   └── feature_builder.py    # assembles the 121-bin x 38-feature frame
│   ├── dataset/
│   │   └── assembler.py          # builds datasets + embargoed walk-forward split
│   ├── models/
│   │   └── lgbm_heatmap.py       # LightGBM per-bin heatmap regressor (V1)
│   ├── eval/
│   │   ├── metrics.py            # peak picking, zone-hit precision/recall, AP/AUC
│   │   ├── baselines.py          # classic S/R baselines as ranked levels
│   │   ├── harness.py            # model-vs-baseline evaluation
│   │   └── plots.py              # heatmap + metric-summary figures
│   └── serving/
│       ├── premarket.py          # premarket map + live IntradayServer engine
│       └── intraday.py           # intraday re-export
├── scripts/
│   ├── run_pipeline.py           # end-to-end demo on synthetic data
│   └── fetch_polygon.py          # same pipeline on real Polygon data
├── outputs/                      # metrics CSVs + JSON + PNG plots land here
├── requirements.txt
└── README.md
```

---

## 13. Model roadmap

**V1 (built): LightGBM per-bin regressor.** One row per (example, bin), two
boosters (support, resistance), cross-entropy objective for the soft targets.
Strong, fast, CPU-only, and interpretable via gain importances. The right
starting point and a genuinely competitive baseline-beater.

**V2 (scaffolded): 1-D CNN / TCN over the price-bin axis.** Treat the 121-bin
feature grid as a 1-D image with 38 channels and convolve along price. This lets
the model learn *shapes* (e.g. "a cluster of touches flanked by volume") and
spatial relationships between bins that a per-bin tree model treats
independently. Same labels, same metrics — a drop-in upgrade of just the model
block.

**V3: sequence model for true streaming intraday.** A small Transformer or
recurrent model over the *time* axis that carries hidden state across bars, so
the intraday map updates incrementally and remembers how levels were respected
earlier in the session rather than recomputing from scratch each bar.

Because the data schema, feature contract, label construction, and evaluation
harness are all model-agnostic, each upgrade swaps only `src/models/` and reuses
everything else.

---

## 14. Using real data

`src/data/polygon_client.py` is a complete, dependency-free (`urllib`-only)
Polygon aggregates client with pagination, retry/backoff, and regular-session
filtering. It maps Polygon's `t/o/h/l/c/v/vw/n` fields to the canonical schema,
so the synthetic and live paths are interchangeable. Set `POLYGON_API_KEY` and
run `scripts/fetch_polygon.py`. (Free tiers are rate-limited and history-capped;
a paid tier is recommended for a real universe.)
