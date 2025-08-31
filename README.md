# FFNet: Feature Fusion Network for Accurate Metabolite Identification

## Overview

**FFNet** is a dual-stream neural network designed for **metabolite identification from mass spectrometry data**. It integrates a **Transformer-based pathway** for capturing global spectral context with a **Multi-Layer Perceptron (MLP) pathway** for extracting local spectral features. These complementary representations are fused through an attentive mechanism to predict comprehensive **molecular fingerprints**, enabling accurate identification of both known and novel metabolites.

---

## Datasets

FFNet has been evaluated on the following datasets:

* **[NIST20](https://chemdata.nist.gov/dokuwiki/doku.php?id=chemdata:start)** – A commercial mass spectral library (cannot be publicly shared; visit website for access)
* **[GNPS](https://gnps.ucsd.edu/)** – Open mass spectrometry data repository
* **[MassBank of North America (MoNA)](https://mona.fiehnlab.ucdavis.edu/)** – Public metabolomics spectral database
* **[CASMI 2022 Dataset](https://fiehnlab.ucdavis.edu/casmi)** – For structure elucidation experiments

> **Note:** Model weights and experimental result data can be downloaded from [Baidu Pan](https://pan.baidu.com/s/16SFB3jjREpdHulcu8Ga3GQ?pwd=dbkw) .

---

## Installation

The required environment can be installed via **conda**:

```bash
# Clone repository
git clone https://github.com/yourusername/FFNet.git

# Create conda environment
conda env create -f environment.yml
conda activate ffnet
```

---


## References

1. [NIST20](https://chemdata.nist.gov/dokuwiki/doku.php?id=chemdata:start)
2. [GNPS](https://gnps.ucsd.edu/)
3. [MoNA](https://mona.fiehnlab.ucdavis.edu/)
4. CASMI 2022: [https://www.casmi-contest.org](https://www.casmi-contest.org)

