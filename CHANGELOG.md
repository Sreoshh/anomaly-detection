# Changelog

## [0.1.0] 

### Added
- Added historical Abohar weather data for 2020–2024.
- Added initial time-series data exploration.
- Added data quality and statistical analysis.
- Added time-based and rolling-window features.
- Added Z-score and IQR anomaly detection baselines.
- Added anomaly visualization for baseline methods.


## [0.1.1] 

### Added
- Added historical Abohar weather data for 2020–2024.
- Added initial timeseries data exploration and visualization.
- Added data quality and statistical analysis.
- Added time-based features including hour, day, month, and year.
- Added rolling 24 hour statistics for temperature, humidity, and pressure.
- Added sensor change and deviation features.
- Added Z-score and IQR anomaly detection baselines.
- Added anomaly visualization for statistical methods.
- Added Isolation Forest anomaly detection.
- Added Isolation Forest anomaly labels and anomaly scores.
- Added analysis of the most anomalous observations.
- Added Isolation Forest anomaly visualization.


## [0.2.0] 

### Added
- Added Isolation Forest anomaly detection with engineered weather features.
- Added Isolation Forest anomaly labels and anomaly scores.
- Added Isolation Forest anomaly visualization.
- Added PyTorch Autoencoder for unsupervised anomaly detection.
- Added chronological train, validation, and test split for Autoencoder training.
- Added feature scaling using StandardScaler.
- Added Autoencoder training and validation loss tracking.
- Added reconstruction error as an anomaly score.
- Added percentile based anomaly thresholding.
- Added Autoencoder anomaly visualization.
- Added comparison of Z-score, IQR, Isolation Forest, and Autoencoder methods.
- Added analysis of anomaly agreement between Isolation Forest and Autoencoder.


## [0.3.0] 

### Added
- Added normalized Isolation Forest and Autoencoder anomaly scores.
- Added combined anomaly score using both ML models.
- Added anomaly severity classification.
- Added unified anomaly results with weather measurements and model flags.
- Added processed weather anomaly score dataset.


## [0.4.0] 

### Added
- Added anomaly temporal pattern analysis.
- Added Isolation Forest and Autoencoder agreement analysis.
- Added anomaly severity distribution analysis.
- Added anomaly visualization for temperature, humidity, and pressure.
- Added evaluation summary for the test dataset.
- Documented unsupervised evaluation approach without ground truth anomaly labels.