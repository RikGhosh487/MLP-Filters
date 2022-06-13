# Photometric Filter Transformations Through MLPs

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](https://github.com/RikGhosh487/MLP-Filters/blob/main/LICENSE) ![Language](https://img.shields.io/badge/language-python-rgb(12%2C%2093%2C%20148)) [![Package](https://img.shields.io/badge/package-pytorch-blueviolet)](https://pytorch.org/) ![Architecture](https://img.shields.io/badge/architecture-MLP-orange)

Due to the lack of a simple conversion from the GAIA photometric magnitude system to the SDSS PSF photometric magnitude system (and vice versa), a pair of Multilayer Perceptrons (MLPs) have been used to find the complex relations between the magnitude systems and regress the magnitude values for one photometric system, by taking the magnitudes from the other photometric system.

Both MLPs have been cross-validated and trained using data collected from the SDSS DR17. To read more about the training data, click [here](https://github.com/RikGhosh487/MLP-Filters/blob/main/data). To refer to the model architecture, utility functions, or the training code itself, click [here](https://github.com/RikGhosh487/MLP-Filters/blob/main/modules).

## Performance
Both models performed really well against the validation dataset and the training dataset.

From SDSS to GAIA\
![Truth-To-Prediction-Plot](https://github.com/RikGhosh487/MLP-Filters/blob/main/Gaia_Perf.png)

The figure above shows the truth to prediction plot for the model's performance when converting sdss magnitudes to gaia magnitudes. The higher the concentration of points along the red-dashed line, the better the overall performance of the model.

| Statistic | Value 
| :---: | :---: 
| Mean G | 0.002924
| Standard Deviation G | 0.026840
| Standard Error G | 0.000920
| RMSE G | 0.026999
| Mean BP | 0.000317
| Standard Deviation BP | 0.035251
| Standard Error BP | 0.001208
| RMSE BP | 0.035253
| Mean RP | -0.003886
| Standard Deviation RP | 0.049786
| Standard Error RP | 0.001706
| RMSE RP | 0.049937

From GAIA to SDSS\
![Truth-To-Prediction-Plot](https://github.com/RikGhosh487/MLP-Filters/blob/main/Sdss_Perf.png)

The figure above shows the truth to prediction plot for the model's performance when converting gaia magnitudes to sdss magnitudes. The higher the concentration of points along the red-dashed line, the better the overall performance of the model.

| Statistic | Value 
| :---: | :---: 
| Mean u | -0.035806
| Standard Deviation u | 0.120327
| Standard Error u | 0.004122
| RMSE u | 0.125542
| Mean g | -0.041327
| Standard Deviation g | 0.056504
| Standard Error g | 0.001936
| RMSE g | 0.070005
| Mean r | -0.045182
| Standard Deviation r | 0.021225
| Standard Error r | 0.000727
| RMSE r | 0.049919
| Mean i | -0.050622
| Standard Deviation i | 0.041739
| Standard Error i | 0.001430
| RMSE i | 0.065610
| Mean z | -0.043206
| Standard Deviation z | 0.062269
| Standard Error z | 0.002133
| RMSE z | 0.075790

## Using
In order to actually use the models, refer to one of the following two `python scripts`:
```
- to_gaia.py -> convert from sdss to gaia
- to_sdss.py -> convert from gaia to sdss
```
Both scripts are designed to handle either reading user input from **STDIN** or by parsing a `.csv` file. For the `.csv` file case, a new `csv` files is created where the computed values are appended to the original `csv` file data.
