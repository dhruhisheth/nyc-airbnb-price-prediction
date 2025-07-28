# Airbnb Price Prediction Model

A machine learning project that predicts Airbnb listing prices in NYC using property features and host characteristics.

## Project Overview

This project implements a complete machine learning pipeline to predict Airbnb listing prices, following the full ML lifecycle from data exploration to model deployment. The goal is to help hosts set competitive rates and enable guests to find accommodations within their budget.

### Problem Statement
- **Dataset**: Airbnb NYC Listings Data (`airbnbListingsData.csv`)
- **Task**: Regression - Predicting listing prices
- **Target Variable**: `price` (continuous numeric value)
- **Problem Type**: Supervised Learning

## Business Value

- **For Hosts**: Set competitive pricing to maximize occupancy and revenue
- **For Guests**: Filter and plan stays within budget constraints
- **For Platform**: Optimize pricing recommendations and market insights

## Dataset Features

### Key Features Used:
- **Property Characteristics**: accommodates, bathrooms, bedrooms, beds, room_type
- **Location**: neighbourhood_group_cleansed
- **Host Information**: host_response_rate, host_acceptance_rate, host_is_superhost
- **Booking Details**: minimum_nights, availability metrics, instant_bookable
- **Reviews**: number_of_reviews, review scores (rating, cleanliness, location, etc.)
- **Calculated Metrics**: reviews_per_month, calculated_host_listings_count

### Data Shape
- **Rows**: 28,022 listings
- **Columns**: 50 features (reduced to 48 after preprocessing)

## Data Preprocessing

### 1. Missing Value Treatment
- **Numeric Features**: Filled with mean values
- **Categorical Features**: Filled with mode values

### 2. Feature Engineering
- Removed irrelevant text columns (names, descriptions, IDs)
- One-hot encoded categorical variables (`neighbourhood_group_cleansed`, `room_type`)
- Converted boolean features to numeric (0/1)

### 3. Feature Scaling
- Applied StandardScaler to all numeric features
- Ensured consistent scale across all input variables

## Model Development

### Models Implemented
1. **Linear Regression** (Baseline)
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**
4. **Hyperparameter-Tuned Random Forest**

### Training Strategy
- **Train/Test Split**: 80/20
- **Cross-Validation**: 5-fold CV for model validation
- **Hyperparameter Tuning**: GridSearchCV for Random Forest optimization

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Primary metric for model comparison
- **MAE (Mean Absolute Error)**: Secondary metric for interpretability

## Project Structure

```
├── data/
│   └── airbnbListingsData.csv
├── Lab8_ML_Project.ipynb
├── README.md
└── requirements.txt
```

## Installation & Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch Jupyter: `jupyter notebook`
4. Open `Lab8_ML_Project.ipynb`
5. Run all cells sequentially

## Key Results

The project demonstrates the complete ML pipeline with model comparison and hyperparameter optimization. Expected outcomes include:

- Baseline Linear Regression performance
- Improved accuracy with ensemble methods (Random Forest, Gradient Boosting)
- Optimized model through hyperparameter tuning
- Feature importance analysis for pricing insights

## Implementation Phases

### Phase 1: Data Understanding
- Exploratory Data Analysis (EDA)
- Missing value assessment
- Feature distribution analysis
- Price distribution visualization

### Phase 2: Data Preparation
- Missing value imputation
- Feature selection and removal
- Categorical encoding
- Feature scaling

### Phase 3: Model Building
- Baseline model implementation
- Ensemble method comparison
- Cross-validation for model selection

### Phase 4: Model Optimization
- Hyperparameter tuning with GridSearchCV
- Performance comparison across models
- Final model selection

### Phase 5: Evaluation
- RMSE and MAE calculation
- Model performance comparison
- Results interpretation

## Technical Specifications

### Libraries Used
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Models**: LinearRegression, RandomForestRegressor, GradientBoostingRegressor
- **Preprocessing**: StandardScaler, train_test_split
- **Evaluation**: cross_val_score, GridSearchCV

### Key Functions
- Data preprocessing pipeline
- Model training and evaluation
- Hyperparameter optimization
- Performance metrics calculation

## Future Enhancements

1. **Advanced Feature Engineering**
   - Text analysis of descriptions and reviews
   - Geographic clustering analysis
   - Seasonal pricing patterns

2. **Model Improvements**
   - XGBoost implementation
   - Neural network approaches
   - Ensemble stacking methods

3. **Deployment Considerations**
   - Model serialization for production
   - API endpoint creation
   - Real-time prediction system

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes as part of a machine learning course assignment.

## Contact

For questions or suggestions regarding this project, please refer to the course materials or instructor guidelines.
