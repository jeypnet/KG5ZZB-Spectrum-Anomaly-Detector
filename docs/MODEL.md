# Spectrum autoencoder: model card

simulator, the baseline or the patch features change.

## What it does
The app compares every new sweep with what it has learned is normal for each frequency bin
(the **baseline**: mean and spread per bin, per hour of the day). The comparison is cut into
**patches**: 64 adjacent bins over the last 4 sweeps. A small neural network, a
**conditional autoencoder**, squeezes each patch through a 16-number bottleneck and rebuilds
it. It was trained on normal patches only, so it rebuilds normal behaviour well (noise,
busy channels switching on and off, slow fading) and anomalies badly. The **anomaly score**
is the rebuild error plus how far the 16-number code lies from the codes of normal patches
(Mahalanobis distance), both scaled to average 1 on normal data. It runs on the CPU inside
the app (ONNX Runtime) and needs no internet.

## Input (384 numbers per patch)
1. **z** (4 x 64): how unusual each bin is, (level - whole-sweep gain offset - usual level) / usual spread,
   with the spread floored at 0.4 dB, compressed as sign(z) ln(1 + |z|) / 2 (near-linear for normal values, strong deviations stay distinct).
2. **spread** (64): log2 of the usual spread over 0.4 dB, clipped to 0-6, divided by 3. Tells the model which bins are normally busy.
3. **above floor** (64): usual level above the local noise floor (20th percentile over 256 bins), clipped to 0-40 dB, divided by 10. Tells it where signals normally are.

The output is one score per patch: `err / mean_err_normal + d2 / 16`, where `err` is the mean
squared error between z and its reconstruction and `d2` the squared Mahalanobis distance of
the bottleneck code from the mean code of the normal training patches (covariance from the
same patches). Why both: the decoder can rebuild some smooth changes, such as a raised noise
floor, quite well; those still push the code to places normal patches never reach. The
"reconstruction error only" row below shows what the second term adds.

## Data
- **Simulator:** . Each scene is a random band of
  768 bins at 2.3-18.8 kHz resolution with 3-21 emitters: steady carriers, digital channels,
  broadcast-width signals, busy channels that switch on/off at random, scheduled
  transmitters (pagers, beacons), slow fading, a rippled noise floor, slow gain wander and
  sweep-to-sweep gain jitter. Each bin's measurement has the spread of an averaged
  periodogram with 23-1875 averages (10-100 ms dwells).
- **Baseline:** learned from 25-120 simulated sweeps per scene with the app's own `Baseline` class.
  A switching emitter that the learning period never saw on (at least twice) is kept off in
  that scene's normal data: from the baseline's point of view it would be a new transmitter.
- **Patches:** built with the app's own `PatchFeatures` code (the same code that runs live).
- **Anomalies (val/test only):** new transmitter (3-25 dB, 3-200 kHz wide), missing signal
  (a steady signal switched off or cut by 6 dB or more), noise floor rise (1-8 dB over 12-35%
  of the band), wideband interference (1-10 dB, 0.3-3 MHz wide), intermittent burst (5-25 dB,
  on in only 1-2 of the 4 sweeps).
- **Splits:** train 72000 normal patches (seed 1, 10% held out for early stopping);
  validation 83932 patches (seed 2, used only to set thresholds); test 83940 patches
  (seed 3: 64000 normal, anomalies per kind [4000, 3940, 4000, 4000, 4000]), never used for training or thresholds.

## Model
Conditional autoencoder, `384 -> 256 -> 64 -> 16` encoder and `16 + 128 -> 128 -> 256 -> 256`
decoder, ReLU, 233,424 parameters. Mean squared error on normal patches, Adam
(lr 1e-3, weight decay 1e-5, halved on plateau), batch 256, early stopping (37 epochs).
Latent mean and covariance are computed on the normal training patches after training.
Exported to ONNX (opset 17) with the whole score computation inside the graph.

## Results on the held-out test set

**In short:** the autoencoder is the best of the four at broad and weak changes (raised
noise floor, wideband interference, anomalies under 3 dB); the per-bin threshold is the best
at strong narrow ones (new carriers, bursts). The **hybrid**, which flags a patch when either
of them does, has the best overall recall and ROC-AUC at the same false-alarm rate, and is the
app's default detector.

### Ranking quality (ROC-AUC, normal vs anomalous patches)
| Detector | ROC-AUC | AUC New transmitter | AUC Missing signal | AUC Noise floor rise | AUC Wideband interference | AUC Intermittent burst |
|---|---|---|---|---|---|---|
| **Hybrid: autoencoder + per-bin** | 0.990 | 0.998 | 0.989 | 0.984 | 0.982 | 0.997 |
| **Autoencoder (neural network)** | 0.932 | 0.938 | 0.933 | 0.990 | 0.990 | 0.809 |
| Autoencoder, reconstruction error only | 0.758 | 0.864 | 0.792 | 0.571 | 0.733 | 0.831 |
| Per-bin threshold (max |z|) | 0.987 | 0.997 | 0.988 | 0.978 | 0.978 | 0.997 |
| Patch energy (mean z^2) | 0.987 | 0.995 | 0.986 | 0.987 | 0.984 | 0.985 |

### At fixed false-alarm rates
Thresholds are set on the **validation** normals; false-alarm rate (FAR), precision and recall
are measured on the **test** set. The app's default is the 0.1% row.
Precision depends on the share of anomalies in the test set (23.8%).

| Target FAR | Detector | Test FAR | Precision | Recall | F1 | New transmitter | Missing signal | Noise floor rise | Wideband interference | Intermittent burst |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1% | Hybrid: autoencoder + per-bin | 0.1% | 0.996 | 0.903 | 0.947 | 94.8% | 88.5% | 81.6% | 90.5% | 96.2% |
| 0.1% | Autoencoder (neural network) | 0.1% | 0.994 | 0.589 | 0.740 | 50.1% | 48.0% | 85.0% | 91.1% | 19.9% |
| 0.1% | Autoencoder, reconstruction error only | 0.1% | 0.992 | 0.361 | 0.530 | 52.4% | 41.2% | 11.7% | 27.2% | 48.2% |
| 0.1% | Per-bin threshold (max |z|) | 0.1% | 0.996 | 0.881 | 0.935 | 96.7% | 90.6% | 65.3% | 89.9% | 97.8% |
| 0.1% | Patch energy (mean z^2) | 0.2% | 0.994 | 0.835 | 0.907 | 87.5% | 81.6% | 84.3% | 91.6% | 72.3% |
| 1.0% | Hybrid: autoencoder + per-bin | 1.1% | 0.965 | 0.954 | 0.960 | 98.6% | 95.5% | 90.9% | 93.2% | 98.9% |
| 1.0% | Autoencoder (neural network) | 1.1% | 0.952 | 0.689 | 0.800 | 65.6% | 61.1% | 91.4% | 93.0% | 33.3% |
| 1.0% | Autoencoder, reconstruction error only | 1.0% | 0.931 | 0.419 | 0.578 | 58.8% | 48.0% | 14.8% | 33.6% | 54.5% |
| 1.0% | Per-bin threshold (max |z|) | 1.1% | 0.963 | 0.943 | 0.953 | 98.8% | 95.6% | 85.2% | 93.2% | 99.0% |
| 1.0% | Patch energy (mean z^2) | 1.1% | 0.963 | 0.938 | 0.950 | 97.0% | 93.5% | 93.2% | 93.6% | 91.6% |

### Recall by anomaly strength (default threshold)
| Strength | Test patches | Hybrid | Autoencoder | Per-bin threshold |
|---|---|---|---|---|
| 0-3 dB | 1930 | 71.0% | 76.7% | 39.1% |
| 3-6 dB | 3816 | 86.5% | 77.8% | 86.6% |
| 6-10 dB | 5306 | 88.9% | 65.3% | 91.3% |
| 10-40 dB | 8888 | 97.0% | 43.1% | 97.4% |

## How the app uses it
- Default detector: **Hybrid (AI + per-bin)**: score = max(autoencoder / 7.997, max|z| / 8.77), unusual above
  1.244 (FAR 0.1% per patch on validation normals). The autoencoder alone uses
  7.997 and the per-bin test alone |z| > 8.77, at the same FAR.
  All thresholds are scaled by the Sensitivity setting.
- Inside flagged patches the app marks the bins that are actually off-normal, groups them,
  and describes each region in plain words. A finding must repeat in 2 sweeps before it becomes an event.

## Scope
- Covers what an RTL-SDR can tune: about 24 MHz to 1.7 GHz, swept in 2 MHz steps.
- Receive-only: the app never transmits.
- Whole-range level changes are removed before scoring (they look the same as a gain change);
  the app reports them separately as a band level shift.
