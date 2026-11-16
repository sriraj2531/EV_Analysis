# Melbourne EV Charging Analysis

**Demand Forecasting, Peak-Load Classification, Station Behaviour
Clustering, and Charging-Duration Forecasting**

This repository contains a Jupyter Notebook analysing hourly
electric-vehicle (EV) charging activity across 63 charging stations in
Melbourne, Australia. The data cover **1 April 2023 to 30 September
2023**.

The project investigates four related questions:

1.  Can hourly charging demand at each station be forecast one hour
    ahead?
2.  Can station-hours with peak charging demand be classified?
3.  Do charging stations form groups with similar usage patterns?
4.  Can charging duration be estimated from historical information?

The analysis is implemented in `Melbourne_EV_Charging_Final.ipynb`.

------------------------------------------------------------------------

## Project objectives

-   Inspect and validate the supplied datasets before modelling.
-   Combine hourly station-level records into a consistent
    station--timestamp dataset.
-   Create calendar variables and station-specific historical lag
    features.
-   Evaluate predictive models using chronological training, validation,
    and test periods.
-   Compare machine-learning methods with simple persistence baselines.
-   Group stations according to their observed charging behaviour.
-   Discuss results, assumptions, and limitations without overstating
    model performance.

## Dataset

The notebook expects the following CSV files:

  -----------------------------------------------------------------------
  File                                Purpose in the notebook
  ----------------------------------- -----------------------------------
  `volume.csv`                        Main hourly station-level demand
                                      variable and target for Task 1

  `duration.csv`                      Charging-duration data and target
                                      for Task 4

  `e_price.csv`                       Electricity-price data; inspected
                                      and used in exploratory analysis

  `s_price.csv`                       Supporting price data; inspected

  `weather.csv`                       City-wide weather observations
                                      joined by timestamp

  `stations.csv`                      Station identifiers, locations, and
                                      infrastructure attributes

  `piles.csv`                         Charger/pile-level information;
                                      inspected but not directly joined

  `distance.csv`                      Pairwise station-distance matrix;
                                      inspected

  `inf.csv`                           Additional station geometry
                                      information; inspected

  `info.csv`                          City-level summary information;
                                      inspected

  `poi.csv`                           Points-of-interest data; inspected
                                      but not spatially aggregated
  -----------------------------------------------------------------------

The five hourly matrices (`volume.csv`, `duration.csv`, `e_price.csv`,
`s_price.csv`, and `weather.csv`) cover 4,392 hourly timestamps. The
station-level matrices contain observations for 63 stations.

**Data note:** The notebook does not establish the physical unit of the
values in `volume.csv` or `duration.csv`. Results are therefore
described in reported volume units and supplied duration units. Confirm
the units from the original dataset documentation before interpreting
them as kWh, minutes, or another physical measure.

### Dataset placement

The notebook currently defines a local data directory in its first code
cell. For a portable GitHub checkout, place the CSV files in a folder
named `data/` beside the notebook and change:

``` python
DATA_FOLDER = Path(r"C:\Users\LENOVO\Downloads\Hourly\MEL")
```

to:

``` python
DATA_FOLDER = Path("data")
```

The expected repository layout is:

``` text
melbourne-ev-charging-analysis/
├── Melbourne_EV_Charging_Final.ipynb
├── README.md
├── requirements.txt
└── data/
    ├── volume.csv
    ├── duration.csv
    ├── e_price.csv
    ├── s_price.csv
    ├── weather.csv
    ├── stations.csv
    ├── piles.csv
    ├── distance.csv
    ├── inf.csv
    ├── info.csv
    └── poi.csv
```

Only upload the dataset files if their licence and distribution terms
permit redistribution. If the dataset cannot be shared publicly, keep
`data/` out of the repository and explain how authorised users can
obtain the files.

------------------------------------------------------------------------

## Methods and workflow

### 1. Data inspection and preparation

All supplied CSV files are loaded and inspected for dimensions, data
types, missing values, duplicates, and sample records. The main hourly
station matrices are reshaped from wide format into long format, with
one record per timestamp and station.

The data are joined using the appropriate keys:

-   Hourly station-level files: `timestamp` and `station_id`.
-   City-wide weather: `timestamp`.
-   Station metadata: `station_id`.

Recorded zero-volume observations are retained because they represent
hours with no recorded charging activity. Files with incompatible
granularity or without a defensible aggregation rule are inspected but
not forced into the modelling table.

### 2. Feature engineering and exploratory analysis

The notebook creates calendar features such as hour, weekday, month, day
of month, and weekend status. Cyclical sine/cosine features represent
the repeating nature of hour-of-day and day-of-week.

Station-specific lag features are created for:

-   Previous hour (`volume_lag_1`)
-   Same hour on the previous day (`volume_lag_24`)
-   Same hour one week earlier (`volume_lag_168`)

Lag features are calculated within each station after sorting by
timestamp. The analysis retains zero demand but removes rows that do not
yet have the required lag history.

Same-hour duration, weather, and price observations are not used as
demand forecast inputs because their availability at the time a forecast
is issued is not established in the notebook.

### 3. Time-aware evaluation

The predictive tasks use chronological splits rather than random row
splitting. Approximately 70% of timestamps are used for training, 15%
for validation, and 15% for testing. All stations belonging to the same
timestamp are assigned to the same subset.

The validation set supports model selection. The test set is held back
for final evaluation. The forecasting setup is a **rolling
one-step-ahead forecast**: the previous hour's actual observation is
assumed to be available before predicting the next hour. It is not a
24-hour-ahead forecast.

### 4. Task 1 --- Demand forecasting

-   Required model: **Linear Regression**
-   Target: station-level hourly reported volume
-   Inputs: historical demand lags, calendar information, and stable
    station metadata
-   Comparisons: previous-hour, previous-day, and previous-week
    persistence baselines; Ridge and Lasso are also examined

The purpose of the baselines is to check whether a more complex model
improves on the simple assumption that the next hour will resemble the
previous hour.

### 5. Task 2 --- Peak-load classification

-   Required model: **Logistic Regression**
-   Target: peak or non-peak station-hour
-   Peak definition: demand at or above the 80th percentile of the
    training-period target values
-   Comparisons: Dummy, Decision Tree, Random Forest, Extra Trees,
    Gradient Boosting, SVM, AdaBoost, and SGD Logistic Regression

Nine candidate classifiers are evaluated using four expanding-window
time-series cross-validation folds within the training period. The
selected probability threshold is determined using validation data. The
previous-hour peak label is included as a benchmark.

### 6. Task 3 --- Station clustering

-   Required model: **K-Means**
-   Unit of analysis: one aggregated behavioural profile per station
-   Features: average demand, maximum demand, demand variability,
    average active-hour duration, active-hour frequency, weekend demand,
    and peak-hour frequency
-   Comparisons: DBSCAN and Gaussian Mixture Model (GMM)

The station features are transformed and standardised before clustering.
Silhouette scores and cluster sizes are used to compare candidate
K-Means values. PCA is used to visualise the station profiles in two
dimensions.

Clustering is descriptive: it groups stations using the observed period
and is not evaluated as a future prediction task. The station with no
recorded demand throughout the period is excluded from behavioural
clustering.

### 7. Task 4 --- Charging-duration forecasting

The notebook predicts the supplied duration value from historical
duration and volume lags, calendar information, and stable station
metadata. Candidate models include Linear Regression, Ridge, Lasso,
Decision Tree, Random Forest, Extra Trees, and Hist Gradient Boosting.
Validation MAE is used for model selection, followed by evaluation on
the untouched test period.

------------------------------------------------------------------------

## Main results

The following results are taken from the notebook's reported test
evaluations.

  -----------------------------------------------------------------------
  Task                    Model                   Test results
  ----------------------- ----------------------- -----------------------
  Demand forecasting      Linear Regression       MAE: **16.8189**; RMSE:
                          (required)              **70.1072**; R²:
                                                  **0.7855**

  Demand baseline         Previous-hour           MAE: **14.2381**; RMSE:
                          persistence             **83.6224**; R²:
                                                  **0.6948**

  Peak classification     Logistic Regression     Accuracy: **0.9177**;
                          (required)              Precision: **0.8613**;
                                                  Recall: **0.8627**; F1:
                                                  **0.8620**

  Peak classification     Gradient Boosting (best Accuracy: **0.9334**;
                          CV candidate)           Precision: **0.8733**;
                                                  Recall: **0.9083**; F1:
                                                  **0.8905**

  Peak baseline           Previous-hour peak      Accuracy: **0.9316**;
                          label                   F1: **0.8852**

  Station clustering      K-Means, K = 2          Silhouette score:
                                                  **0.4198**; cluster
                                                  sizes: **32 and 30
                                                  stations**

  Duration forecasting    Extra Trees             MAE: **0.3453**; RMSE:
                                                  **0.6081**; R²:
                                                  **0.9127**

  Duration baseline       Previous-hour           MAE: **0.2895**; RMSE:
                          persistence             **0.6803**; R²:
                                                  **0.8908**
  -----------------------------------------------------------------------

### Interpretation

-   Historical charging activity is strongly informative about the next
    hour.
-   Linear Regression has better demand RMSE and R² than the
    previous-hour baseline, but the baseline has lower MAE. The learned
    model is therefore not better on every metric.
-   Gradient Boosting achieves the best F1 among the compared learned
    classifiers and modestly exceeds the previous-hour peak baseline.
-   K-Means separates the stations into lower-demand and higher-demand
    groups.
-   Extra Trees explains a high proportion of duration variation, but
    the previous-hour duration baseline has lower MAE.
-   Results must be interpreted alongside the baseline comparisons and
    the dataset limitations.

------------------------------------------------------------------------

## How to run the project

### Requirements

-   Python 3.9 or later is recommended.
-   Jupyter Notebook or JupyterLab.
-   The Python packages listed in `requirements.txt`.

### Setup

1.  Clone or download this repository.

2.  Create and activate a Python virtual environment (recommended).

3.  Install the dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

4.  Place the authorised dataset CSV files in the `data/` directory.

5.  Update `DATA_FOLDER` in the first notebook code cell to
    `Path("data")`.

6.  Open `Melbourne_EV_Charging_Final.ipynb` in Jupyter Notebook or
    JupyterLab.

7.  Run the notebook cells from top to bottom.

The notebook creates a `melbourne_project_outputs/` directory in the
current working directory for generated output files, where applicable.

### Reproducibility

The notebook uses fixed random states for stochastic modelling steps
where specified. Results can still vary with package versions, platform
differences, or changes to the source data. The chronological split and
the source CSV files should remain unchanged when reproducing the
reported results.

------------------------------------------------------------------------

## Limitations

1.  The dataset covers six months in one city; results may not
    generalise to other seasons, years, or locations.
2.  The physical units of volume and duration are not established in the
    notebook.
3.  The weather file contains missing observations.
4.  Some station time series contain unusually smooth, repeated, or
    constant values and should be checked against the original data
    provider.
5.  A single global peak threshold can favour stations with larger
    demand scales.
6.  The predictive tasks evaluate rolling one-step-ahead forecasts, not
    multi-hour operational forecasts.
7.  Pooled error metrics can be strongly influenced by stations with
    very large demand.
8.  Clustering summarises behaviour over the observed period and does
    not establish that station groups will remain stable in the future.

------------------------------------------------------------------------

## Repository contents

-   `Melbourne_EV_Charging_Final.ipynb` --- complete analysis,
    modelling, evaluation, and visualisations.
-   `README.md` --- project overview, methods, results, and
    instructions.
-   `requirements.txt` --- Python dependencies.
-   `data/` --- source CSV files, if redistribution is permitted.

## Acknowledgement and data provenance

The analysis is based on the supplied CHARGED hourly Melbourne (MEL)
dataset files. Add the dataset's original publication, author, licence,
and download URL here if those details are available from the source
documentation. Do not claim ownership of the underlying dataset unless
that is accurate.

## Authors

Add the project author(s), course, institution, and contact information
here.

## License

Add an appropriate repository licence after confirming the rights to
distribute the notebook and any included data. A code licence does not
automatically grant permission to redistribute third-party datasets.
