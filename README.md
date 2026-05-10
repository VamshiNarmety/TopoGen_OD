# TopoGen-OD: Synthetic Urban Mobility and Network Evaluation

TopoGen-OD builds hourly origin-destination (OD) data, trains baseline and generative models, and checks whether the predictions preserve the transport network structure.

## What this repo covers

- OD construction from raw NYC and Chicago taxi trips
- Network building, centrality, community detection, and disruption analysis
- Baseline models: Historical Mean, gravity, radiation, OD marginal, community-aware, centrality-aware, and MLP
- Generative models: cVAE, cGAN, topology-aware cVAE, and intervention-aware cVAE
- Post-training plots for accuracy, topology, temporal variation, and spatial hotspots

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/VamshiNarmety/TopoGen_OD.git
cd TopoGen_OD
```

### 2. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Complete End-to-End Pipeline

### Full NYC Dataset (2023)

#### Step 1a: Download NYC data
```bash
source .venv/bin/activate

python src/data/download_data.py \
  --dataset yellow \
  --start-month 2023-01 \
  --end-month 2023-12
```

#### Step 1b: Download Chicago data
```bash
python src/data/download_data.py \
  --dataset chicago \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --chicago-page-size 50000
```

**Outputs:** 
- `data/raw/trip_data/yellow/` (12 NYC monthly parquets)
- `data/raw/trip_data/chicago/` (Chicago chunked parquets)

---

### Step 2: Build OD Matrices (full datasets)

#### Step 2a: NYC OD matrix
```bash
python src/data/build_od.py \
  --input-glob 'data/raw/trip_data/yellow/*.parquet' \
  --provider nyc_yellow \
  --output-dir data/processed/od \
  --prefix hourly_od_nyc_2023_full
```

#### Step 2b: Chicago OD matrix
```bash
python src/data/build_od.py \
  --input-glob 'data/raw/trip_data/chicago/*.parquet' \
  --provider chicago_taxi \
  --output-dir data/processed/od \
  --prefix hourly_od_chicago_2023_full
```

**Outputs:** Hourly OD aggregates (pickup_hour, origin, destination, trip_count)
- `data/processed/od/hourly_od_nyc_2023_full.parquet`
- `data/processed/od/hourly_od_chicago_2023_full.parquet`

---

### Step 3: Build Transport Graph (for both cities)

#### Step 3a: NYC graph
```bash
python src/network/build_graph.py \
  --input data/processed/od/hourly_od_nyc_2023_full.parquet \
  --output-dir data/processed/network \
  --prefix network_nyc_2023_full
```

#### Step 3b: Chicago graph
```bash
python src/network/build_graph.py \
  --input data/processed/od/hourly_od_chicago_2023_full.parquet \
  --output-dir data/processed/network \
  --prefix network_chicago_2023_full
```

**Outputs:** Weighted directed graphs
- `data/processed/network/network_nyc_2023_full.graphml`
- `data/processed/network/network_chicago_2023_full.graphml`

---

### Step 4: Network Analysis (for both cities)

#### Step 4a: NYC Network Analysis
```bash
# Compute baseline metrics
python src/network/metrics.py \
  --input data/processed/network/network_nyc_2023_full_edges.parquet \
  --output-dir data/processed/network \
  --prefix network_nyc_2023_full

# Centrality analysis
python src/network/centrality_analysis.py \
  --input data/processed/network/network_nyc_2023_full_edges.parquet \
  --output-dir data/processed/network/centrality \
  --prefix centrality_nyc_2023_full

# Community detection
python src/network/community_detection.py \
  --input data/processed/network/network_nyc_2023_full_edges.parquet \
  --output-dir data/processed/network/community \
  --prefix community_nyc_2023_full

# Disruption simulation
python src/network/disruption_simulation.py \
  --input data/processed/network/network_nyc_2023_full_edges.parquet \
  --centrality data/processed/network/centrality/centrality_nyc_2023_full_centrality.parquet \
  --output-dir data/processed/network/disruption \
  --prefix disruption_nyc_2023_full
```

#### Step 4b: Chicago Network Analysis
```bash
# Compute baseline metrics
python src/network/metrics.py \
  --input data/processed/network/network_chicago_2023_full_edges.parquet \
  --output-dir data/processed/network \
  --prefix network_chicago_2023_full

# Centrality analysis
python src/network/centrality_analysis.py \
  --input data/processed/network/network_chicago_2023_full_edges.parquet \
  --output-dir data/processed/network/centrality \
  --prefix centrality_chicago_2023_full

# Community detection
python src/network/community_detection.py \
  --input data/processed/network/network_chicago_2023_full_edges.parquet \
  --output-dir data/processed/network/community \
  --prefix community_chicago_2023_full

# Disruption simulation
python src/network/disruption_simulation.py \
  --input data/processed/network/network_chicago_2023_full_edges.parquet \
  --centrality data/processed/network/centrality/centrality_chicago_2023_full_centrality.parquet \
  --output-dir data/processed/network/disruption \
  --prefix disruption_chicago_2023_full
```

#### Step 5a: NYC Baselines
```bash
python src/models/train_baselines.py \
  --input data/processed/od/hourly_od_nyc_2023_full.parquet \
  --output-dir data/processed/baselines \
  --prefix baseline_nyc_2023_full_normal \
  --epochs 50 \
  --batch-size 4096
```

#### Step 5b: Chicago Baselines
```bash
python src/models/train_baselines.py \
  --input data/processed/od/hourly_od_chicago_2023_full.parquet \
  --output-dir data/processed/baselines \
  --prefix baseline_chicago_2023_full_normal \
  --epochs 50 \
  --batch-size 4096
```

**Baseline suite includes:**
- Historical Mean
- OD Marginal (production × attraction)
- Gravity model
- Radiation model
- Community-aware baseline
- Centrality-aware baseline
- MLP

**Latest 50-epoch baseline results:**

| Model | NYC MAE / RMSE | Chicago MAE / RMSE |
|---|---:|---:|
| Historical Mean | 1.0740 / 2.0066 | 1.3162 / 4.2202 |
| OD Marginal | 2.0278 / 3.4747 | 2.5545 / 6.4757 |
| Gravity | 1.7408 / 3.9139 | 2.7376 / 9.5021 |
| Radiation | 2.7808 / 4.7032 | 3.6236 / 9.9850 |
| Community-aware | 1.7146 / 3.6801 | 2.8048 / 9.6640 |
| Centrality-aware | 1.7685 / 3.9777 | 2.8076 / 9.6911 |
| MLP | 1.1039 / 2.1612 | 1.3544 / 4.3666 |

### Step 6: Train Deep Generative Baselines (50 epochs, full datasets)

These are trained separately from `train_baselines.py`.

#### Step 6a: NYC Deep Generative Models
```bash
python src/models/train_deepgen_baselines.py \
  --input data/processed/od/hourly_od_nyc_2023_full.parquet \
  --output-dir data/processed/deepgen_baselines \
  --prefix deepgen_nyc_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

#### Step 6b: Chicago Deep Generative Models
```bash
python src/models/train_deepgen_baselines.py \
  --input data/processed/od/hourly_od_chicago_2023_full.parquet \
  --output-dir data/processed/deepgen_baselines \
  --prefix deepgen_chicago_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

**Models included:**
- Plain cVAE baseline (non-topology-constrained)
- cGAN baseline

**Outputs per city:**
- `deepgen_<city>_2023_full_epochs50_plain_cvae_metrics.json`
- `deepgen_<city>_2023_full_epochs50_cgan_metrics.json`
- `deepgen_<city>_2023_full_epochs50_comparison.csv`
- `deepgen_<city>_2023_full_epochs50_test_predictions.parquet`

**Latest 50-epoch deep generative results:**

| Model | NYC MAE / RMSE | Chicago MAE / RMSE |
|---|---:|---:|
| Plain cVAE | 1.4620 / 2.8115 | 2.2889 / 7.8945 |
| cGAN | 1.0983 / 2.1790 | 1.3247 / 4.3240 |

### Step 7: Train Novelty Model — Topology-constrained cVAE (50 epochs)

#### Step 7a: NYC Topology-aware Model
```bash
python src/models/train_topology_cvae.py \
  --input data/processed/od/hourly_od_nyc_2023_full.parquet \
  --output-dir data/processed/novelty \
  --prefix novelty_topocvae_nyc_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

#### Step 7b: Chicago Topology-aware Model
```bash
python src/models/train_topology_cvae.py \
  --input data/processed/od/hourly_od_chicago_2023_full.parquet \
  --output-dir data/processed/novelty \
  --prefix novelty_topocvae_chicago_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

**Novelty idea:**
- cVAE conditioned on OD + time + topology features
- Topology-prior consistency regularization (community + centrality features)
- Graph Laplacian smoothness on learned zone embeddings

**Outputs per city:**
- `novelty_topocvae_<city>_2023_full_epochs50_topocvae_metrics.json`
- `novelty_topocvae_<city>_2023_full_epochs50_comparison.csv`
- `novelty_topocvae_<city>_2023_full_epochs50_test_predictions.parquet`

**Latest 50-epoch topology-aware results:**

| Model | NYC MAE / RMSE | Chicago MAE / RMSE |
|---|---:|---:|
| Topology-constrained cVAE | 1.4507 / 3.1539 | 2.3371 / 8.2899 |
| Topology prior | 1.7065 / 3.7921 | 2.8051 / 9.6774 |

### Step 8: Train Intervention-aware Novelty Model (50 epochs)

Explicit intervention conditioning for counterfactual OD flow redistribution under network disruptions.

#### Step 8a: NYC Intervention Model
```bash
python src/models/train_intervention_topocvae.py \
  --input data/processed/od/hourly_od_nyc_2023_full.parquet \
  --output-dir data/processed/novelty \
  --prefix novelty_intervention_topocvae_nyc_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

#### Step 8b: Chicago Intervention Model
```bash
python src/models/train_intervention_topocvae.py \
  --input data/processed/od/hourly_od_chicago_2023_full.parquet \
  --output-dir data/processed/novelty \
  --prefix novelty_intervention_topocvae_chicago_2023_full_epochs50 \
  --epochs 50 \
  --batch-size 4096
```

**Intervention scenarios:**
- none (factual)
- edge_removed (link disruption)
- hub_removed (high-degree node removal)
- node_added (new zone introduction)

**Outputs per city:**
- `novelty_intervention_topocvae_<city>_2023_full_epochs50_metrics.json`
- `novelty_intervention_topocvae_<city>_2023_full_epochs50_comparison.csv`
- `novelty_intervention_topocvae_<city>_2023_full_epochs50_scenario_summary.json`
- `novelty_intervention_topocvae_<city>_2023_full_epochs50_test_predictions.parquet`

**Latest 50-epoch intervention-aware results:**

| Model | NYC MAE / RMSE | Chicago MAE / RMSE |
|---|---:|---:|
| Intervention-aware cVAE (none) | 1.4996 / 3.2461 | 2.3825 / 8.4449 |
| Topology prior | 1.7065 / 3.7921 | 2.8051 / 9.6774 |

### Step 9: Generate Visualizations

After all models are trained, generate comprehensive evidence plots (comparisons, topology preservation, temporal patterns, spatial hotspots, disruption analysis):

#### Step 9a: NYC Visualizations
```bash
python src/models/plot_experiment_evidence.py \
  --output-dir data/processed/figures/nyc_full_final \
  --prefix nyc_full_final \
  --baseline-comp data/processed/baselines/baseline_nyc_2023_full_normal_comparison.csv \
  --deepgen-comp data/processed/deepgen_baselines/deepgen_nyc_2023_full_epochs50_comparison.csv \
  --novelty-comp data/processed/novelty/novelty_topocvae_nyc_2023_full_epochs50_comparison.csv \
  --intervention-comp data/processed/novelty/novelty_intervention_topocvae_nyc_2023_full_epochs50_comparison.csv \
  --disruption-comp data/processed/network/disruption/disruption_nyc_2023_full_comparison.csv \
  --mlp-metrics data/processed/baselines/baseline_nyc_2023_full_normal_mlp_metrics.json \
  --deepgen-cvae-metrics data/processed/deepgen_baselines/deepgen_nyc_2023_full_epochs50_plain_cvae_metrics.json \
  --deepgen-cgan-metrics data/processed/deepgen_baselines/deepgen_nyc_2023_full_epochs50_cgan_metrics.json \
  --novelty-metrics data/processed/novelty/novelty_topocvae_nyc_2023_full_epochs50_topocvae_metrics.json \
  --intervention-metrics data/processed/novelty/novelty_intervention_topocvae_nyc_2023_full_epochs50_metrics.json \
  --baseline-preds data/processed/baselines/baseline_nyc_2023_full_normal_test_predictions.parquet \
  --deepgen-preds data/processed/deepgen_baselines/deepgen_nyc_2023_full_epochs50_test_predictions.parquet \
  --novelty-preds data/processed/novelty/novelty_topocvae_nyc_2023_full_epochs50_test_predictions.parquet \
  --intervention-preds data/processed/novelty/novelty_intervention_topocvae_nyc_2023_full_epochs50_test_predictions.parquet \
  --graph-topk 150 \
  --graph-max-models 8 \
  --graph-max-nodes 60 \
  --graph-include-models topocvae,none,edge_removed,hub_removed,node_added,topology_prior,cgan,mlp \
  --spatial-zone-shapefile data/raw/misc/taxi_zones.shp \
  --spatial-top-zones 50 \
  --spatial-max-models 6 \
  --spatial-include-models topocvae,none,edge_removed,hub_removed,node_added,topology_prior
```

#### Step 9b: Chicago Visualizations
```bash
python src/models/plot_experiment_evidence.py \
  --output-dir data/processed/figures/chicago_full_final \
  --prefix chicago_full_final \
  --baseline-comp data/processed/baselines/baseline_chicago_2023_full_normal_comparison.csv \
  --deepgen-comp data/processed/deepgen_baselines/deepgen_chicago_2023_full_epochs50_comparison.csv \
  --novelty-comp data/processed/novelty/novelty_topocvae_chicago_2023_full_epochs50_comparison.csv \
  --intervention-comp data/processed/novelty/novelty_intervention_topocvae_chicago_2023_full_epochs50_comparison.csv \
  --disruption-comp data/processed/network/disruption/disruption_chicago_2023_full_comparison.csv \
  --mlp-metrics data/processed/baselines/baseline_chicago_2023_full_normal_mlp_metrics.json \
  --deepgen-cvae-metrics data/processed/deepgen_baselines/deepgen_chicago_2023_full_epochs50_plain_cvae_metrics.json \
  --deepgen-cgan-metrics data/processed/deepgen_baselines/deepgen_chicago_2023_full_epochs50_cgan_metrics.json \
  --novelty-metrics data/processed/novelty/novelty_topocvae_chicago_2023_full_epochs50_topocvae_metrics.json \
  --intervention-metrics data/processed/novelty/novelty_intervention_topocvae_chicago_2023_full_epochs50_metrics.json \
  --baseline-preds data/processed/baselines/baseline_chicago_2023_full_normal_test_predictions.parquet \
  --deepgen-preds data/processed/deepgen_baselines/deepgen_chicago_2023_full_epochs50_test_predictions.parquet \
  --novelty-preds data/processed/novelty/novelty_topocvae_chicago_2023_full_epochs50_test_predictions.parquet \
  --intervention-preds data/processed/novelty/novelty_intervention_topocvae_chicago_2023_full_epochs50_test_predictions.parquet \
  --graph-topk 140 \
  --graph-max-models 8 \
  --graph-max-nodes 55 \
  --graph-include-models topocvae,none,edge_removed,hub_removed,node_added,topology_prior,cgan,mlp \
  --spatial-zone-shapefile data/raw/misc/chicago_community_areas.geojson \
  --spatial-top-zones 60 \
  --spatial-max-models 6 \
  --spatial-include-models topocvae,none,edge_removed,hub_removed,node_added,topology_prior
```

**Outputs per city:**
- Model comparison plots (MAE, RMSE, by hour, by day)
- Training curves and convergence analysis
- Model-wise temporal variation (hourly patterns)
- Error distributions (prediction vs actual)
- Prediction scatter plots
- Topology preservation scorecards
- Actual vs predicted graph topology grids
- Spatial hotspot maps (top zones by MAE)
- Network disruption impact analysis
- Summary JSON with all metrics
## Repository Structure

```
TopoGen_OD/
├── README.md                          # This file (complete end-to-end pipeline)
├── requirements.txt                   # Python package dependencies (PyTorch + core ML stack)
├── .gitignore                         # Git exclusions
│
├── src/                               # Source code
│   ├── data/
│   │   ├── download_data.py           # Download NYC TLC + Chicago taxi data
│   │   └── build_od.py                # Convert raw trips → hourly OD matrices
│   │
│   ├── network/
│   │   ├── build_graph.py             # OD matrices → weighted directed graphs
│   │   ├── metrics.py                 # Compute degree, strength, assortativity
│   │   ├── centrality_analysis.py     # 6 centrality measures (betweenness, closeness, etc.)
│   │   ├── community_detection.py     # Louvain community detection
│   │   ├── disruption_simulation.py   # Generate network disruption scenarios
│   │   └── plot_disruption_results.py # Disruption impact visualizations
│   │
│   └── models/
│       ├── train_baselines.py         # Historical Mean + gravity/radiation + MLP
│       ├── train_deepgen_baselines.py # cVAE + cGAN baselines
│       ├── train_topology_cvae.py     # Topology-constrained cVAE (novelty 1)
│       ├── train_intervention_topocvae.py  # Intervention-aware topology cVAE (novelty 2)
│       └── plot_experiment_evidence.py # Post-training visualization engine
│
└── data/
    ├── raw/                           # Original downloaded data (git-ignored)
    │   ├── trip_data/
    │   │   ├── yellow/                # NYC TLC yellow taxi parquets
    │   │   └── chicago/               # Chicago taxi parquets
    │   └── misc/
    │       ├── taxi_zones/            # NYC taxi zone shapefiles
    │       └── chicago_community_areas.geojson  # Chicago zone boundaries
    │
    └── processed/                     # Pipeline outputs (git-ignored)
        ├── od/                        # Hourly OD matrices
        ├── network/                   # Graph + centrality + community + disruption
        ├── baselines/                 # Baseline model outputs & checkpoints
        ├── deepgen_baselines/         # cVAE & cGAN outputs
        ├── novelty/                   # Topology-constrained & intervention model outputs
        └── figures/                   # Final visualization artifacts
            ├── nyc_full_final/        # NYC comparison plots, topology grids, hotspots
            └── chicago_full_final/    # Chicago comparison plots, topology grids, hotspots
```

---

## Data Flow Diagram

```
Raw taxi trips (NYC + Chicago, 2023)
       ↓
  [download_data.py] → Parquet files per month
       ↓
  [build_od.py] → Hourly OD matrices (trips per hour, zone-to-zone)
       ↓
  [build_graph.py] → Weighted directed graphs (nodes=zones, edges=flows)
       ↓
  ├→ [metrics.py] → Network statistics (degree, strength, assortativity)
  ├→ [centrality_analysis.py] → Node importance (6 measures)
  ├→ [community_detection.py] → Zone communities (Louvain algorithm)
  └→ [disruption_simulation.py] → Resilience scenarios (15 variants: node/edge removal)
       ↓
  [train_baselines.py] → 7 baseline predictors (historical, gravity, MLP, etc.)
       ↓
  [train_deepgen_baselines.py] → cVAE + cGAN deep generative models
       ↓
  [train_topology_cvae.py] → Topology-constrained cVAE (novelty model 1)
       ↓
  [train_intervention_topocvae.py] → Intervention-aware cVAE (novelty model 2)
       ↓
  [plot_experiment_evidence.py] → Comprehensive evidence visualization
       │
       ├→ Model comparison (MAE, RMSE by hour/day)
       ├→ Topology preservation (graph structure preservation)
       ├→ Temporal patterns (hourly variation, day-of-week)
       ├→ Spatial hotspots (top zones by error)
       ├→ Network disruption impact (resilience analysis)
       └→ Training curves & convergence
```

---

## Key Output Artifacts

| File | Purpose | Key Metrics |
|------|---------|-----------|
| `hourly_od_nyc_2023_full.parquet` | NYC OD demand matrix | 8.7M rows, 263 zones |
| `hourly_od_chicago_2023_full.parquet` | Chicago OD demand matrix | 2.1M rows, 77 zones |
| `network_<city>_2023_full.graphml` | Transport network graph | Weighted directed graph |
| `disruption_<city>_2023_full_comparison.csv` | Network resilience | 15 disruption scenarios |
| `baseline_<city>_2023_full_normal_comparison.csv` | 7 baseline predictions | MAE / RMSE per model |
| `deepgen_<city>_2023_full_epochs50_comparison.csv` | Deep generative models | cVAE & cGAN performance |
| `novelty_topocvae_<city>_2023_full_epochs50_comparison.csv` | Topology-aware novelty | Topology preservation scores |
| `novelty_intervention_<city>_2023_full_epochs50_comparison.csv` | Intervention novelty | Counterfactual accuracy |
| `<city>_full_final_evidence_summary.json` | All metrics summary | Complete model evaluation |
| `<city>_full_final_*.png` | Visualization artifacts | 10+ figures per city |

---
