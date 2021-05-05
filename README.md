# Online Aggregation of Probability Forecasts with Confidence (APFC)

This repository contains code for paper Online Aggregation of Probability Forecasts with Confidence.

We use implementation details described in the paper.

## Implementation of the method

APFC folder contains all the Matlab scripts required to execute the method:

APLF.m is the main file.

initialize.m function inizializes model parameters.

prediction.m function obtain load forecasts and
probabilistic load forecasts
in form of aggregated probability distribution function.

test.m function quantifies the prediction errors RMSE and
MAPE.

update_model.m function updates the model for each new
training sample.

update_parameters.m function updates model parameters.

## Data

We use publicly available dataset used in GEFCom2014
Global Energy Forecasting Competition 2014 dataset (Load treck).

We save the data in .mat files that contain a structures with following fields:

Hourly load time series

Temperature time series

Date and hour or timestamp when the load is measure

Table of synchronized calendar parameters (seasons, days of week, parts of day)

## Installation
