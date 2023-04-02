# Disentangling the rhythms of human activity in the built environment for airborne transmission risk
This repository provides the data and source code for the following study: Zachary Susswein, Eva Rest, Shweta Bansal. "Disentangling the rhythms of human activity in the built environment for airborne transmission risk". https://doi.org/10.1101/2022.04.07.22273578 (currently in peer review)

## Deriving the metric
In the `deriving_indoor_activity_metric` directory, we provide the code for deriving the indoor activity metric using the Safegraph Weekly Patterns data (which are available for researchers directly from Safegraph). We also provide the file of indoor/outdoor classifications that we derived for each Safegraph POI.

## Data
Data on the indoor activity seasonality metric (\sigma) that we define in this work can be found in the `indoor_activity_data` directory. The `indoor_activity_2018_2020.csv` file contains weekly, county-level (fips) indoor activity estimates.

## Analysis
#### Time series clustering
Python code for the time series clustering in this work can be found in the `time_series_clustering` directory.

#### Sinusoidal fits
R code for the sinusoidal fits in this work can be found in the `sinusoidal_fit` directory.
