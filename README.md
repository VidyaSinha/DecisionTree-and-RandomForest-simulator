🧩 Decision Tree and Random Forest Simulator
This is an interactive Decision Tree and Random Forest simulation tool where you can visualize model behavior in real-time using any dataset. This tool allows you to fine-tune hyperparameters, select features, and observe how these machine learning algorithms create decision boundaries.

📊 Key Features
Upload Your Dataset: Supports CSV datasets with ease.
Real-time Visualization: Get real-time visualizations of Decision Tree and Random Forest structures.
Hyperparameter Tuning: Adjust parameters like split criteria, max depth, and tree splitters.
Interactive and User-Friendly: Powered by Streamlit for a smooth user experience.

Demo
[https://youtu.be/x9HUv53ctfc]

📥 Sample Dataset
A sample wine quality dataset has been provided for testing. The dataset is sourced from Kaggle.
Alternatively, you can use any CSV dataset of your choice.

📈 Hyperparameters Explained
Split Criteria:

poisson: For count-based regression tasks.
absolute_error: Optimizes based on mean absolute error.
squared_error: Default for regression (mean squared error).
friedman_mse: An improvement of MSE for better splits.
Splitter:

best: Chooses the best possible split.
random: Chooses a random split.
Max Depth: Limits the depth of the decision tree.

How It Works:-
Select Dataset: Upload your CSV file.
Choose Features: Select the feature columns and target variable.
Adjust Parameters: Tune the hyperparameters to see how models adapt.
Visualize Models: View the Decision Tree and Random Forest models in real-time.

🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
Libraries: Scikit-learn, Pandas, Matplotlib

## Deployment Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the app using Streamlit:
   ```
   streamlit run app.py
   ```

Use Cases
Understanding Decision Trees and Random Forest models.
Learning how hyperparameters affect machine learning models.
Testing performance of different datasets in a visualized environment.
