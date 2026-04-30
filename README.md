# Terra-TPU: Earth Observation Segmentation with Google Cloud TPUs & Keras Kinetic

This repository presents a modular and scalable architecture for training Deep Learning models applied to **Earth Observation (EO)**, using **Aquaculture** mapping in satellite imagery (Landsat) as a case study.

The core differentiator of this project is its integration with the [`keras-team/kinetic`](https://github.com/keras-team/kinetic) library. Kinetic enables the seamless training of complex models (like U-Nets) on **Google Cloud TPUs (v5e / v6e)** by abstracting the infrastructure complexity of Kubernetes (GKE).

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