# Terra-TPU: Earth Observation Segmentation with Google Cloud TPUs & Keras Kinetic

This repository presents a modular and scalable architecture for training Deep Learning models applied to **Earth Observation (EO)**, using **Aquaculture** mapping in satellite imagery  as a case study.

The core differentiator of this project is its integration with the [`keras-team/kinetic`](https://github.com/keras-team/kinetic) library. Kinetic enables the seamless training of complex models (like U-Nets) on **Google Cloud TPUs (v5e / v6e)** by abstracting the infrastructure complexity of Kubernetes (GKE).

![Shrimp ponds (left) and salmon pens, as viewed from satellites © Dynaspace](shrimp-ponds-salmon-pens-from-satellite-credit-dynaspace.avif)
**Shrimp ponds (left) and salmon pens, as viewed from satellites © Dynaspace**
---

## Application: Aquaculture Mapping

The dataset utilized in this repository consists of multispectral Landsat imagery prepared for the detection of aquaculture ponds. The implemented `U-Net Model` leverages a hybrid loss function (`binary_crossentropy` + `IoU`) optimized to handle the severe class imbalance inherent in segmenting small, structured water bodies across vast landscapes.

---

## Key Features

* **Modular Architecture:** Strict separation of concerns between the data pipeline (`tf.data`), model architecture, and execution script, ensuring high cohesion and maintainability.
* **Serverless-like TPU Execution:** Utilizing the `@kinetic.run` decorator to package and dispatch training routines directly to GKE clusters, eliminating the need to write YAML manifests or configure Dockerfiles manually.
* **Configuration-Driven Reproducibility:** Infrastructure scaling and hardware topologies are controlled entirely via environment variables (`.env`).
* **Next-Gen Hardware Support:** Built-in support for Google's TPU v5 Lite Podslices (v5e) and v6e topologies, leveraging the JAX/TensorFlow ecosystem for maximum performance.

---

## Repository Structure

The project has been carefully refactored to ensure that heavy library *imports* occur only in the remote TPU environment, optimizing payload delivery.
```text
Terra-TPU/
├── data_loader.py       # Data pipeline and preprocessing logic (tf.data.Dataset, TFRecords)
├── model.py             # Neural network architecture definition (e.g., Keras U-Net)
├── train_kinetic.py     # Main script featuring the @kinetic.run decorator for orchestration
├── requirements.txt     # Python dependencies for both local and remote environments
├── run.sh               # Helper shell script for deployment automation
└── .env.example         # Template for environment variables
```

---

## Prerequisites

- Python 3.10+
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) installed and configured
- A GCP project with billing enabled
- TPU Quota: Granted quota for TPU v5 Lite Podslice or v6e chips in your target region. [Consult availability beforehand](https://docs.cloud.google.com/tpu/docs/regions-zones#north-america)
- Data Access: A Google Cloud Storage (GCS) bucket containing the dataset. (Note: The training and validation data for this project were originally generated and exported from Google Earth Engine directly into GCS in TFRecord format).

## GCP & Credential Setup

1. Authenticate with Google Cloud

Authenticate your local machine to allow Kinetic to provision resources and access the GCS bucket:
```bash

gcloud auth login
gcloud auth application-default login
```

2. Select Your GCP Project

```bash

# List your existing projects
gcloud projects list

# Set the active project
gcloud config set project YOUR_PROJECT_ID
```

3. Configure Environment Variables

The entire infrastructure and data pipeline are controlled via environment variables.
```bash
cp .env.example .env
```
Edit the .env file with your specific infrastructure and data paths:

```bash
KINETIC_PROJECT="your-gcp-project-id"
KINETIC_ZONE="us-central1-a"
KINETIC_CLUSTER="kinetic-earth-cluster"
TPU_TYPE="v5litepod-1" # Or v6e-8 for a 2x4 topology
```
# Data Configuration (Earth Engine -> GCS Pipeline)
```bash
BUCKET_NAME="your-earth-engine-exports-bucket"
TRAIN_DATA_PATH="lulc_samples/train"
VAL_DATA_PATH="lulc_samples/eval"
```

## Quick Start
```bash
# 1. Clone & install dependencies
git clone https://github.com/MariaLuize/Terra-TPU.git
cd Terra-TPU
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Source your environment variables
source .env

# 3. Provision TPU infrastructure (one-time setup, ~5 mins)
# Note: Kinetic will use Pulumi to build a GKE cluster with your selected TPU
kinetic up --project=$KINETIC_PROJECT --zone=$KINETIC_ZONE --cluster=$KINETIC_CLUSTER --yes

# 4. Dispatch the remote training job to the TPU
# This will build the container, send the context, and stream logs back
./run.sh

# 5. Clean up (Important — avoids idle GKE and TPU costs!)
kinetic down --project=$KINETIC_PROJECT --zone=$KINETIC_ZONE --cluster=$KINETIC_CLUSTE
```



