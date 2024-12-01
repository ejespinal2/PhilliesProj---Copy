# Projected OBP - Machine Learning Project

## Overview

This project leverages machine learning to predict **On-Base Percentage (OBP)** for baseball players in the 2021 season using historical player statistics. By applying advanced data processing and Random Forest regression, this project offers an insightful approach to analyzing and forecasting player performance.

## Features

- **Weighted OBP Calculation**: Implements a custom calculation based on historical plate appearances and OBP, inspired by Marcel Forecasting methods.
- **Machine Learning Model**: Utilizes a Random Forest Regressor for robust prediction of OBP values while accounting for outliers.
- **Visualization and Reporting**: Generates comprehensive visualizations, including residual and predicted vs. actual plots, for performance evaluation.
- **Exported Results**: Outputs player statistics and predictions to a CSV file for further analysis.

## Key Files

- **`obp.csv`**: Input dataset containing player statistics from 2016 to 2020, including plate appearances (PA) and OBP.
- **`main.py`**: Core script for data processing, model training, prediction, and visualization.
- **`player_obp_comparison.csv`**: Output file containing original statistics, predictions, and errors for players.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ejespinal2/ProjectedOBP-MachineLearningProject.git
   ```
2. Install required Python libraries:
   ```bash
   pip install pandas scikit-learn matplotlib numpy
   ```

## Usage

1. Place the dataset `obp.csv` in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Visualizations will be saved as PNG files, and a comparison table will be generated as `player_obp_comparison.csv`.

## Results

- **Mean Squared Error (MSE)**: Evaluates the accuracy of the model.
- **Residual Plot**: Visualizes the prediction error.
- **Predicted vs Actual Plot**: Compares model predictions against observed values.

## References

- Weighted OBP calculations are inspired by Marcel Forecasting: [Triples Alley Blog](https://triplesalley.wordpress.com/2010/12/22/marcel-and-forecasting-systems/).
- Visualization techniques adapted from community discussions on [Stack Overflow](https://stackoverflow.com/questions/49992300/python-how-to-show-graph-in-visual-studio-code-itself).

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Author

Developed by [E.J. Espinal](https://github.com/ejespinal2).

---
