import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Upload CSV dataset
st.title("Decision Trees and Random Forests")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Display the dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select feature columns and target column
    feature_columns = st.multiselect("Select Feature Columns", df.columns.tolist())
    target_column = st.selectbox("Select Target Column", df.columns.tolist())

    # Check if task is classification or regression
    is_classification = df[target_column].dtype == 'object' or len(df[target_column].unique()) < 20

    # Sidebar options for hyperparameters
    st.sidebar.title("Model Hyperparameters")

    # Decision Tree Hyperparameters
    st.sidebar.subheader("Decision Tree Hyperparameters")
    dt_criterion = st.sidebar.selectbox("Criterion (Decision Tree)", ["squared_error", "friedman_mse", "absolute_error", "poisson"] if not is_classification else ["gini", "entropy"])
    dt_splitter = st.sidebar.selectbox("Splitter (Decision Tree)", ("best", "random"))
    dt_min_samples_split = st.sidebar.slider("Min Samples Split (Decision Tree)", 2, 10, 2)
    dt_min_samples_leaf = st.sidebar.slider("Min Samples Leaf (Decision Tree)", 1, 5, 1)
    dt_max_depth = st.sidebar.slider("Max Depth (Decision Tree)", 1, 10, 3)

    # Random Forest Hyperparameters
    st.sidebar.subheader("Random Forest Hyperparameters")
    rf_criterion = st.sidebar.selectbox("Criterion (Random Forest)", ["squared_error", "friedman_mse", "absolute_error", "poisson"] if not is_classification else ["gini", "entropy"])
    rf_n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)
    rf_max_features = st.sidebar.selectbox("Max Features (Random Forest)", ["sqrt", "log2", None])
    rf_min_samples_split = st.sidebar.slider("Min Samples Split (Random Forest)", 2, 10, 2)
    rf_min_samples_leaf = st.sidebar.slider("Min Samples Leaf (Random Forest)", 1, 5, 1)
    rf_max_depth = st.sidebar.slider("Max Depth (Random Forest)", 1, 10, 3)

    # Prepare feature and target data
    X = df[feature_columns].values
    y = df[target_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Display options for model selection
    st.sidebar.subheader("Choose Model")
    model_choice = st.sidebar.radio("Select Model", ["Decision Tree", "Random Forest"])

    if model_choice == "Decision Tree":
        # Build Decision Tree model
        if is_classification:
            model = DecisionTreeClassifier(
                criterion=dt_criterion,
                splitter=dt_splitter,
                min_samples_split=dt_min_samples_split,
                min_samples_leaf=dt_min_samples_leaf,
                max_depth=dt_max_depth,
                random_state=42
            )
        else:
            model = DecisionTreeRegressor(
                criterion=dt_criterion,
                splitter=dt_splitter,
                min_samples_split=dt_min_samples_split,
                min_samples_leaf=dt_min_samples_leaf,
                max_depth=dt_max_depth,
                random_state=42
            )

        # Fit the model
        model.fit(X_train, y_train)

        # Predict and calculate performance
        y_pred = model.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Decision Tree Performance")
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.subheader("Decision Tree Performance")
            st.write(f"Mean Squared Error: {mse:.2f}")

        # Display Decision Tree visualization
        st.subheader("Decision Tree Visualization")
        plt.figure(figsize=(15, 10))
        plot_tree(model, filled=True, feature_names=feature_columns, class_names=np.unique(y_train).astype(str), rounded=True)
        st.pyplot(plt.gcf())

    elif model_choice == "Random Forest":
        # Build Random Forest model
        if is_classification:
            model = RandomForestClassifier(
                criterion=rf_criterion,
                n_estimators=rf_n_estimators,
                max_features=rf_max_features,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                max_depth=rf_max_depth,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                criterion=rf_criterion,
                n_estimators=rf_n_estimators,
                max_features=rf_max_features,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                max_depth=rf_max_depth,
                random_state=42
            )

        # Fit the model
        model.fit(X_train, y_train)

        # Predict and calculate performance
        y_pred = model.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Random Forest Performance")
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
        else:
            mse = mean_squared_error(y_test, y_pred)
            st.subheader("Random Forest Performance")
            st.write(f"Mean Squared Error: {mse:.2f}")

        # Feature importance
        st.subheader("Feature Importance")
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": feature_importances})
        st.write(importance_df.sort_values(by="Importance", ascending=False))

        # Visualize one of the trees in the Random Forest
        tree_index = st.slider("Select Tree for Visualization", 0, len(model.estimators_) - 1, 0)
        st.subheader(f"Visualization of Tree {tree_index} in the Random Forest")
        plt.figure(figsize=(15, 10))
        plot_tree(model.estimators_[tree_index], filled=True, feature_names=feature_columns, class_names=np.unique(y_train).astype(str), rounded=True)
        st.pyplot(plt.gcf())

    # Explanation of Hyperparameters
    st.subheader("Understanding Hyperparameters")
    st.write("""
    **Decision Tree:**
    - **Criterion**: The function to measure the quality of a split. For classification, options are 'gini' or 'entropy'. For regression, options include 'squared_error' and others.
    - **Splitter**: Strategy to split at each node ('best' or 'random').
    - **Min Samples Split**: Minimum number of samples required to split an internal node.
    - **Min Samples Leaf**: Minimum number of samples required to be at a leaf node.
    - **Max Depth**: Maximum depth of the tree.

    **Random Forest:**
    - **Criterion**: Same as Decision Tree but applied across multiple trees.
    - **Number of Trees (n_estimators)**: Total trees in the forest. More trees reduce overfitting but increase computation.
    - **Max Features**: Controls the number of features considered for the best split ('sqrt', 'log2', or None).
    - **Min Samples Split and Leaf**: Similar to Decision Tree but applied across all trees.
    - **Max Depth**: Limits the depth of each tree to prevent overfitting.
    """)

    # Understanding Model Behavior
    st.subheader("Model Behavior")
    st.write("""
    - **Overfitting**: When the model performs well on training data but poorly on unseen data.
    - **Underfitting**: When the model is too simple to capture the data's patterns.
    - **Feature Importance**: Shows the contribution of each feature to the prediction in Random Forest.
    - **Tree Visualization**: Helps understand the decision-making process of individual trees.
    """)
