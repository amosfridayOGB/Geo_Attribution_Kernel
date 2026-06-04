# Geo-Attribution Kernel

A high-performance, vectorized geospatial multi-touch attribution (MTA) engine optimized via NumPy arrays for large-scale transaction probability analysis. This framework algorithmically maps digital marketing touchpoints against spatial data to calculate decay-adjusted intent proximity scores across mass-scale user journeys.

---

## 🔬 Core Methodology & Mathematical Formulation

Traditional attribution models ignore physical proximity and spatial friction. This engine bridges that gap by modeling user touchpoints as a multi-dimensional matrix where conversion intent is a function of both channel-specific weights and spatial decay.

### Spatial Decay Function
The spatial friction of a touchpoint diminishes exponentially based on the user's distance from a target geographical anchor. The localized score $S$ for a given touchpoint distance $d$ (in kilometers) is modeled using an exponential decay function:

$$S(d) = e^{-\lambda d}$$

Where:
* $d$ is the Euclidean or Geodetic distance of the user touchpoint from the target coordinate asset.
* $\lambda$ is the spatial decay constant, scaling the velocity of intent attrition over physical distance.

### Cumulative Attribution Matrix
For a mass-scale array of user journeys containing $N$ touchpoints, the cumulative system matrix maps conversion probability via vectorized dot products, eliminating row-by-row iteration bottlenecks:

$$A_{\text{score}} = \sum_{i=1}^{N} \left( W_{\text{channel}} \cdot S(d_i) \cdot I_{\text{count}} \right)$$

---

## ⚡ Performance Architecture

* **Pure Vectorization:** Built entirely on top of optimized contiguous memory blocks using standard NumPy arrays, mapping arrays of $(10000, 3)$ shapes smoothly with zero structural latency.
* **Deterministic Simulating Pipeline:** Employs an independent, repeatable data generation pipeline to stress-test the attribution kernel against synthetic high-density lead distributions.
* **Asymmetric Channel Calibration:** Pre-configured to track complex, modern multi-channel conversion funnels including:
  * Top-of-Funnel (ToF): `meta_awareness_ad`
  * Middle-of-Funnel (MoF): `meta_conversion_ad`
  * Bottom-of-Funnel (BoF): `whatsapp_lead_trigger`

---

## 📂 Project Structure

```text
├── core_logic/
│   ├── analyzed_metrics.csv       # Excluded via .gitignore (Mass-scale output data)
│   └── simulated_touchpoints.csv  # Excluded via .gitignore (Raw simulation data)
├── .gitignore                     # Production-grade Git ignore filters
├── generate_data.py               # Deterministic structural data simulation script
├── kernel.py                      # Main vectorized matrix attribution engine
└── README.md                      # Academic & structural documentation
