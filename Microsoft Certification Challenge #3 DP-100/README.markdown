# Gelato Mágico: Ice Cream Sales Prediction 🍦

This project uses Machine Learning to predict ice cream sales based on daily temperature for Gelato Mágico, a fictional ice cream shop. By leveraging a Linear Regression model, MLflow for experiment tracking, and a structured pipeline, the project aims to optimize production, reduce waste, and maximize profits.

## Project Structure
- `inputs/ice_cream_sales.csv`: Synthetic dataset with temperature and sales data.
- `train_model.py`: Python script for training and logging the model with MLflow.
- `README.md`: Project documentation.

## Dataset
The dataset (`inputs/ice_cream_sales.csv`) contains 100 rows with:
- `temperature`: Daily temperature in Celsius (15°C to 35°C).
- `sales`: Number of ice creams sold (linearly correlated with temperature plus noise).

Example:
```
temperature,sales
15.2,152.3
16.8,168.7
...
```

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn mlflow
   ```
3. **Run the training script**:
   ```bash
   python train_model.py
   ```
4. **View MLflow UI**:
   ```bash
   mlflow ui
   ```
   Open `http://localhost:5000` in your browser to see experiment results.

## Process
1. **Data Preparation**: Loaded the dataset and split it into training (80%) and test (20%) sets.
2. **Pipeline**: Created a scikit-learn pipeline with:
   - `StandardScaler`: To normalize the temperature feature.
   - `LinearRegression`: To predict sales.
3. **Training**: Trained the model and evaluated it using Mean Squared Error (MSE) and R².
4. **MLflow**: Logged parameters (`test_size`, `random_state`), metrics (`mse`, `r2`), and the model.
5. **Results**: The model achieved a low MSE and high R², indicating good predictive performance.

## Insights
- **Linear Relationship**: The synthetic data shows a strong linear correlation between temperature and sales, making Linear Regression a suitable choice.
- **MLflow Benefits**: MLflow made it easy to track experiments, compare metrics, and reproduce results, which is critical for real-world ML projects.
- **Scalability**: The pipeline ensures reproducibility and can be extended with more features (e.g., day of the week, holidays).
- **Cloud Potential**: Deploying this model on Azure ML could enable real-time predictions via APIs, helping the shop adjust production daily.

## Possibilities for Improvement
- **Additional Features**: Include factors like humidity, day of the week, or holidays to improve predictions.
- **Advanced Models**: Try polynomial regression or tree-based models if the relationship is non-linear.
- **Real-Time Deployment**: Use Azure ML to deploy the model as an endpoint for real-time predictions.
- **Data Collection**: Replace synthetic data with real sales data for better accuracy.

## Technologies Used
- Python, pandas, numpy
- scikit-learn (LinearRegression, Pipeline, StandardScaler)
- MLflow
- GitHub for version control

## Conclusion
This project demonstrates a practical application of Machine Learning to solve a business problem. By predicting ice cream sales, Gelato Mágico can optimize production and reduce waste. The use of MLflow and a pipeline ensures reproducibility, making this a robust solution for real-world deployment.

Happy coding, and enjoy some ice cream! 🍦