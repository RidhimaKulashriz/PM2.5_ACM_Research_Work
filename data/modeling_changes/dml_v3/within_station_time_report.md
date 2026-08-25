# Within-Station Expanding Time DML

> This is a sensitivity design. Station means are learned from earlier years only and applied to later holdouts; no holdout outcome or treatment is used in the transformation.

The design scores 968 rows across 3 annual holdouts. The within-station estimate is -44.397641 with station-clustered 95% interval [-66.307516, -22.487766].

Outcome nuisance RMSE/R² are 27.782535/0.843337; treatment nuisance RMSE/R² are 0.059111/0.241912.

This design removes training-period station means but does not solve time-varying confounding, measurement error, spatial spillovers, or simultaneity. It is not a replacement for the primary station-grouped DML estimate.
