# Training data contract

A production dataset should preserve what was knowable at each historical prediction time. Never overwrite an old forecast with a later revision.

## Required columns

- `target_time`: ISO-8601 time for the value being predicted.
- `actual_value`: official observed value aligned with the contract's settlement station.
- `forecast_mean`: forecast value available at the historical prediction time.
- `station_code`: exact station identifier.
- `market_type`: such as `daily_high_temperature`.

## Recommended features

`latitude`, `longitude`, `elevation_m`, `lead_hours`, `forecast_median`, `forecast_std`, `forecast_min`, `forecast_max`, `latest_temperature`, `observation_trend_1h`, `dew_point`, `humidity`, `wind_speed`, `cloud_cover`, `pressure`, `precipitation_probability`, `model_count`, and `data_age_seconds`.

## Historical evaluation fields

For event-probability evaluation, include `lower_bound`, `upper_bound`, and `market_yes_ask`. Bounds use an inclusive lower edge and exclusive upper edge.

The built-in Open-Meteo ingestion route is intended for bootstrap experimentation. Before relying on the model operationally, replace or augment its gridded labels with the exact official-station observations and timestamped forecast archives relevant to each contract.
