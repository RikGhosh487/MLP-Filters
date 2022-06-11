# Training and Validation Data

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://github.com/RikGhosh487/Open-Cluster/blob/main/LICENSE) ![Format](https://img.shields.io/badge/format-.csv-rgb(12%2C%2093%2C%20148))

This **directory** contains two subdirectories, each of which contains a CSV file obtained from the SQL [query](https://github.com/RikGhosh487/MLP-Filters/blob/main/data_extraction.sql). The entire data (train and validation combined) can be found in `mlp.csv`.

The train dataset has a size of 4822 individual observations. This dataset contains values from the **5** SDSS PSF filters and **3** GAIA filters. The same training data is used to train both models (`GAIA` → `SDSS` and `SDSS` → `GAIA`).

The valid dataset has a size of 852 individual observations. This dataset also contains values from the **5** SDSS PSF filters and **3** GAIA filters. The same validation data is used to validate both models (`GAIA` → `SDSS` and `SDSS` → `GAIA`).
