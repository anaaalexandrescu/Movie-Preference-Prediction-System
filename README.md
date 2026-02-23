# Movie Preference Prediction System

## 1. Overview
This project implements a machine learning pipeline designed to predict whether a user will like a specific movie. By leveraging historical rating data, user demographics, and movie metadata, the system frames the recommendation problem as a binary classification task. It calculates comprehensive profiles for both users and items to accurately forecast user preferences.

## 2. Pipeline Overview
The workflow consists of several interconnected stages:
* Data Ingestion and Exploratory Data Analysis (EDA)
* Target Definition and Class Imbalance Checking
* Advanced Feature Engineering (User/Movie Profiling)
* Data Preprocessing and Scaling
* Model Training (Random Forest and XGBoost)
* Performance Evaluation and Visualization

## 3. Key Features and Technical Implementation

### Data Processing and Target Framing
The system processes three interconnected datasets: user demographics, movie metadata (including genres and release dates), and historical user-item interaction logs (ratings).
* *Binary Target Creation*: Instead of predicting exact ratings on a scale of 1 to 5, the problem is formulated as predicting a "Like." A rating of 4 or 5 is strictly classified as a positive class (1 - Liked), while ratings of 3 and below are classified as the negative class (0 - Not Liked).
* *Outlier Detection*: The pipeline identifies users with abnormally high or low rating activities and movies with outlier popularity using the Interquartile Range (IQR) method during the exploratory phase.

### Advanced Feature Engineering
To enable the machine learning models to understand user tastes, several complex features are engineered from the raw relational data:
* *User and Movie Statistical Profiles*: The system aggregates historical data to compute the mean, standard deviation, and total count of ratings for every single user and movie. This helps the model understand if a user is generally a harsh critic or if a movie is universally acclaimed.
* *Rating Difference*: A calculated feature representing the absolute difference between a user's historical average rating and a movie's overall average rating.
* *Historical Genre Preferences*: For every user, the system calculates the percentage of their past ratings that belong to each specific genre (e.g., what percentage of their watched movies were 'Action' or 'Comedy').
* *Genre Match Score*: This is a powerful interaction feature. It calculates a personalized compatibility score by multiplying the user's historical genre preference percentages with the binary genre flags of the target movie. It essentially quantifies how well a specific movie aligns with the user's established genre tastes.
* *Demographic Integration*: User age and encoded gender are integrated into the final feature matrix to account for demographic trends in movie preferences.

### Preprocessing and Dimensionality
* *Sanitization*: The engineered feature matrix is rigorously checked and sanitized to replace any infinite (Inf) or Not-a-Number (NaN) values with zeros, ensuring mathematical stability during model training.
* *Standardization*: All features are standardized using a StandardScaler so that they possess a mean of zero and a standard deviation of one. This prevents features with larger numerical ranges (like 'total number of ratings') from disproportionately influencing the models.

### Modeling and Evaluation
The predictive engine relies on two robust ensemble learning algorithms:
* *Random Forest Classifier*: Utilizes a bagging approach, building 100 decision trees to establish a highly interpretable baseline with controlled maximum depth to prevent overfitting.
* *XGBoost Classifier*: A high-performance gradient boosting framework. It builds trees sequentially, correcting errors made by previous iterations. It uses a specific learning rate, column subsampling, and row subsampling to achieve strong generalization.
* *Stratified Validation*: The train-test split is stratified based on the target variable to ensure that both the training and testing sets maintain the same proportion of "Liked" vs "Not Liked" instances.
* *Comprehensive Metrics*: The models are evaluated using multiple metrics: Accuracy, Precision, Recall, and the F1-Score (which balances Precision and Recall). Cross-validation is also performed to guarantee the model's stability across different subsets of data. 
* *Visual Artifacts*: The pipeline automatically generates and exports high-resolution comparative plots, including distribution histograms, side-by-side Confusion Matrices, and top 15 Feature Importance charts for both algorithms.
