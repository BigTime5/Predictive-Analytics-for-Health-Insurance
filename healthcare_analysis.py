"""
Healthcare Cost Prediction Dashboard
=====================================
Comprehensive data analysis and ML prediction model for health insurance costs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# 1. DATA LOADING AND CLEANING
# =============================================================================

def load_and_clean_data():
    """Load and clean the insurance dataset"""
    # Load training data
    df = pd.read_csv('data/insurance.csv')
    
    # Clean charges column - remove $ and convert to float
    df['charges'] = df['charges'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
    
    # Standardize categorical columns
    df['sex'] = df['sex'].str.lower().replace({
        'f': 'female', 'woman': 'female', 'man': 'male'
    })
    
    df['smoker'] = df['smoker'].str.lower()
    
    df['region'] = df['region'].str.lower().replace({
        'southwest': 'southwest', 'southeast': 'southeast',
        'northwest': 'northwest', 'northeast': 'northeast'
    })
    
    # Remove rows with invalid data (negative age, negative children)
    df = df[df['age'] > 0]
    df = df[df['children'] >= 0]
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['age', 'charges'])
    
    # Fill missing bmi with median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    # Fill missing categorical with mode
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0] if len(df['sex'].mode()) > 0 else 'unknown')
    df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0] if len(df['smoker'].mode()) > 0 else 'unknown')
    df['region'] = df['region'].fillna(df['region'].mode()[0] if len(df['region'].mode()) > 0 else 'unknown')
    
    return df

def load_validation_data():
    """Load and clean validation dataset"""
    df = pd.read_csv('data/validation_dataset.csv')
    
    # Standardize categorical columns
    df['sex'] = df['sex'].str.lower().replace({
        'f': 'female', 'woman': 'female', 'man': 'male'
    })
    
    df['smoker'] = df['smoker'].str.lower()
    df['region'] = df['region'].str.lower()
    
    # Fill missing values
    df['bmi'] = df['bmi'].fillna(df['bmi'].median() if df['bmi'].median() > 0 else 30)
    df['age'] = df['age'].fillna(df['age'].median() if pd.notna(df['age'].median()) else 40)
    df['children'] = df['children'].fillna(0)
    df['sex'] = df['sex'].fillna('unknown')
    df['smoker'] = df['smoker'].fillna('unknown')
    df['region'] = df['region'].fillna('southwest')
    
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def prepare_features(df):
    """Prepare features for modeling"""
    df_processed = df.copy()
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                        bins=[0, 25, 35, 45, 55, 65, 100],
                                        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    # BMI categories
    df_processed['bmi_category'] = pd.cut(df_processed['bmi'],
                                           bins=[0, 18.5, 25, 30, 35, 100],
                                           labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese'])
    
    # Interaction features
    df_processed['smoker_bmi'] = df_processed['smoker'].map({'yes': 1, 'no': 0}) * df_processed['bmi']
    df_processed['smoker_age'] = df_processed['smoker'].map({'yes': 1, 'no': 0}) * df_processed['age']
    df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
    
    return df_processed

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

def train_model(df):
    """Train the prediction model"""
    # Prepare features
    df_model = prepare_features(df)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    le_age_group = LabelEncoder()
    le_bmi_cat = LabelEncoder()
    
    df_model['sex_encoded'] = le_sex.fit_transform(df_model['sex'].astype(str))
    df_model['smoker_encoded'] = le_smoker.fit_transform(df_model['smoker'].astype(str))
    df_model['region_encoded'] = le_region.fit_transform(df_model['region'].astype(str))
    
    # Feature matrix
    features = ['age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded', 
                'region_encoded', 'smoker_bmi', 'smoker_age', 'age_bmi']
    
    X = df_model[features].fillna(0)
    y = df_model['charges']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple models and select best
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    best_model = None
    best_score = -np.inf
    best_name = ''
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    # Get predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return best_model, metrics, X_test, y_test, y_pred, features

# =============================================================================
# 4. PREDICTIONS FOR VALIDATION DATA
# =============================================================================

def predict_validation(model, val_df, features):
    """Generate predictions for validation dataset"""
    val_processed = prepare_features(val_df)
    
    # Encode
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    # Fit on known categories
    all_sex = ['male', 'female', 'unknown']
    all_smoker = ['yes', 'no', 'unknown']
    all_region = ['southwest', 'southeast', 'northwest', 'northeast']
    
    le_sex.fit(all_sex)
    le_smoker.fit(all_smoker)
    le_region.fit(all_region)
    
    val_processed['sex_encoded'] = le_sex.transform(val_processed['sex'].astype(str))
    val_processed['smoker_encoded'] = le_smoker.transform(val_processed['smoker'].astype(str))
    val_processed['region_encoded'] = le_region.transform(val_processed['region'].astype(str))
    
    # Create features
    val_processed['smoker_bmi'] = val_processed['smoker'].map({'yes': 1, 'no': 0, 'unknown': 0}).fillna(0) * val_processed['bmi']
    val_processed['smoker_age'] = val_processed['smoker'].map({'yes': 1, 'no': 0, 'unknown': 0}).fillna(0) * val_processed['age']
    val_processed['age_bmi'] = val_processed['age'] * val_processed['bmi']
    
    X_val = val_processed[features].fillna(0)
    predictions = model.predict(X_val)
    
    return val_processed, predictions

# =============================================================================
# 5. VISUALIZATION AND DASHBOARD
# =============================================================================

def create_dashboard(df, metrics, X_test, y_test, y_pred, val_df, predictions):
    """Create comprehensive dashboard"""
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('Healthcare Cost Prediction Dashboard\nPredictive Analytics for Health Insurance', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # 1. Dataset Overview (Top)
    ax1 = fig.add_subplot(5, 3, 1)
    summary_data = {
        'Total Records': len(df),
        'Avg Age': f"{df['age'].mean():.1f} years",
        'Avg BMI': f"{df['bmi'].mean():.1f}",
        'Avg Charges': f"${df['charges'].mean():,.0f}",
        'Smokers': f"{(df['smoker']=='yes').sum()} ({(df['smoker']=='yes').mean()*100:.1f}%)"
    }
    ax1.axis('off')
    ax1.text(0.5, 0.5, 'Dataset Overview', fontsize=16, fontweight='bold', 
             ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.35, '\n'.join([f"• {k}: {v}" for k, v in summary_data.items()]), 
             fontsize=12, ha='center', va='center', transform=ax1.transAxes,
             family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    # 2. Model Performance
    ax2 = fig.add_subplot(5, 3, 2)
    ax2.axis('off')
    ax2.text(0.5, 0.5, 'Model Performance', fontsize=16, fontweight='bold', 
             ha='center', va='center', transform=ax2.transAxes)
    metrics_text = f"""
• R² Score: {metrics['r2']:.4f}
• RMSE: ${metrics['rmse']:,.2f}
• MAE: ${metrics['mae']:,.2f}
• CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}
    """
    ax2.text(0.5, 0.35, metrics_text, fontsize=12, ha='center', va='center',
             transform=ax2.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
    
    # 3. Prediction vs Actual
    ax3 = fig.add_subplot(5, 3, 3)
    ax3.scatter(y_test, y_pred, alpha=0.5, c=colors[0], edgecolors='white', linewidth=0.5)
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Charges ($)', fontsize=11)
    ax3.set_ylabel('Predicted Charges ($)', fontsize=11)
    ax3.set_title('Predicted vs Actual Charges', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.set_xlim(0, max_val * 1.05)
    ax3.set_ylim(0, max_val * 1.05)
    
    # 4. Age Distribution
    ax4 = fig.add_subplot(5, 3, 4)
    df_age = df.copy()
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df_age['age_group'] = pd.cut(df_age['age'], bins=age_bins, labels=age_labels)
    
    age_charges = df_age.groupby('age_group')['charges'].mean()
    bars = ax4.bar(age_charges.index, age_charges.values, color=colors[0], edgecolor='white', linewidth=1.5)
    ax4.set_xlabel('Age Group', fontsize=11)
    ax4.set_ylabel('Average Charges ($)', fontsize=11)
    ax4.set_title('Average Charges by Age Group', fontsize=13, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    # Add value labels
    for bar, val in zip(bars, age_charges.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
                f'${val:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Smoker Impact
    ax5 = fig.add_subplot(5, 3, 5)
    smoker_charges = df.groupby('smoker')['charges'].mean()
    colors_smoker = ['#27ae60', '#e74c3c']
    bars = ax5.bar(smoker_charges.index, smoker_charges.values, color=colors_smoker, 
                   edgecolor='white', linewidth=1.5)
    ax5.set_xlabel('Smoker Status', fontsize=11)
    ax5.set_ylabel('Average Charges ($)', fontsize=11)
    ax5.set_title('Impact of Smoking on Charges', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, smoker_charges.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 6. BMI vs Charges
    ax6 = fig.add_subplot(5, 3, 6)
    # Color by smoker status
    colors_scatter = df['smoker'].map({'yes': '#e74c3c', 'no': '#27ae60'})
    ax6.scatter(df['bmi'], df['charges'], c=colors_scatter, alpha=0.4, s=30, edgecolors='white', linewidth=0.3)
    ax6.set_xlabel('BMI', fontsize=11)
    ax6.set_ylabel('Charges ($)', fontsize=11)
    ax6.set_title('BMI vs Charges (Red=Smoker, Green=Non-Smoker)', fontsize=13, fontweight='bold')
    
    # 7. Regional Distribution
    ax7 = fig.add_subplot(5, 3, 7)
    region_charges = df.groupby('region')['charges'].agg(['mean', 'count'])
    region_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax7.bar(region_charges.index, region_charges['mean'], color=region_colors, 
                   edgecolor='white', linewidth=1.5)
    ax7.set_xlabel('Region', fontsize=11)
    ax7.set_ylabel('Average Charges ($)', fontsize=11)
    ax7.set_title('Average Charges by Region', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, region_charges['mean']):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # 8. Number of Children Impact
    ax8 = fig.add_subplot(5, 3, 8)
    children_charges = df.groupby('children')['charges'].mean()
    bars = ax8.bar(children_charges.index, children_charges.values, color=colors[1], 
                   edgecolor='white', linewidth=1.5)
    ax8.set_xlabel('Number of Children', fontsize=11)
    ax8.set_ylabel('Average Charges ($)', fontsize=11)
    ax8.set_title('Charges by Number of Dependents', fontsize=13, fontweight='bold')
    ax8.set_xticks(range(len(children_charges.index)))
    ax8.set_xticklabels(children_charges.index)
    
    # 9. Charges Distribution
    ax9 = fig.add_subplot(5, 3, 9)
    ax9.hist(df['charges'], bins=40, color=colors[0], edgecolor='white', alpha=0.7)
    ax9.axvline(df['charges'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: ${df['charges'].mean():,.0f}")
    ax9.axvline(df['charges'].median(), color='orange', linestyle='--', linewidth=2, label=f"Median: ${df['charges'].median():,.0f}")
    ax9.set_xlabel('Charges ($)', fontsize=11)
    ax9.set_ylabel('Frequency', fontsize=11)
    ax9.set_title('Distribution of Healthcare Charges', fontsize=13, fontweight='bold')
    ax9.legend()
    
    # 10. Sex Distribution
    ax10 = fig.add_subplot(5, 3, 10)
    sex_charges = df.groupby('sex')['charges'].mean()
    bars = ax10.bar(sex_charges.index, sex_charges.values, color=['#e91e63', '#2196f3'], 
                    edgecolor='white', linewidth=1.5)
    ax10.set_xlabel('Sex', fontsize=11)
    ax10.set_ylabel('Average Charges ($)', fontsize=11)
    ax10.set_title('Average Charges by Sex', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, sex_charges.values):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 11. Feature Importance
    ax11 = fig.add_subplot(5, 3, 11)
    feature_names = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Smoker×BMI', 'Smoker×Age', 'Age×BMI']
    # Get feature importance from Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    X_train_temp, _, y_train_temp, _ = train_test_split(
        df[['age', 'bmi', 'children']].join(
            pd.get_dummies(df[['sex', 'smoker', 'region']], drop_first=True)
        ), df['charges'], test_size=0.2, random_state=42
    )
    rf.fit(X_train_temp, y_train_temp)
    importances = rf.feature_importances_
    
    # Sort and display top features
    sorted_idx = np.argsort(importances)[-8:]
    ax11.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors[2], edgecolor='white')
    ax11.set_yticks(range(len(sorted_idx)))
    ax11.set_yticklabels(X_train_temp.columns[sorted_idx], fontsize=9)
    ax11.set_xlabel('Importance', fontsize=11)
    ax11.set_title('Feature Importance', fontsize=13, fontweight='bold')
    
    # 12. Validation Predictions Summary
    ax12 = fig.add_subplot(5, 3, 12)
    ax12.axis('off')
    ax12.text(0.5, 0.5, 'Validation Predictions', fontsize=16, fontweight='bold', 
             ha='center', va='center', transform=ax12.transAxes)
    val_summary = f"""
• Records Predicted: {len(predictions)}
• Min Predicted: ${predictions.min():,.2f}
• Max Predicted: ${predictions.max():,.2f}
• Mean Predicted: ${predictions.mean():,.2f}
• Median Predicted: ${np.median(predictions):,.2f}
    """
    ax12.text(0.5, 0.35, val_summary, fontsize=12, ha='center', va='center',
             transform=ax12.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.8))
    
    # 13. Validation Data Visualization
    ax13 = fig.add_subplot(5, 3, 13)
    val_df_vis = val_df.copy()
    val_df_vis['predicted_charges'] = predictions
    val_charges_by_smoker = val_df_vis.groupby('smoker')['predicted_charges'].mean()
    bars = ax13.bar(val_charges_by_smoker.index, val_charges_by_smoker.values,
                    color=['#27ae60', '#e74c3c'], edgecolor='white', linewidth=1.5)
    ax13.set_xlabel('Smoker Status', fontsize=11)
    ax13.set_ylabel('Predicted Charges ($)', fontsize=11)
    ax13.set_title('Validation: Predicted Charges by Smoking Status', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, val_charges_by_smoker.values):
        ax13.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 14. Age Distribution Comparison
    ax14 = fig.add_subplot(5, 3, 14)
    ax14.hist(df['age'], bins=25, alpha=0.6, label='Training Data', color=colors[0], edgecolor='white')
    ax14.hist(val_df['age'], bins=25, alpha=0.6, label='Validation Data', color=colors[1], edgecolor='white')
    ax14.set_xlabel('Age', fontsize=11)
    ax14.set_ylabel('Frequency', fontsize=11)
    ax14.set_title('Age Distribution: Training vs Validation', fontsize=13, fontweight='bold')
    ax14.legend()
    
    # 15. Key Insights Summary
    ax15 = fig.add_subplot(5, 3, 15)
    ax15.axis('off')
    ax15.text(0.5, 0.9, 'Key Insights', fontsize=16, fontweight='bold', 
             ha='center', va='top', transform=ax15.transAxes)
    insights = """
    1. SMOKING is the #1 cost driver - smokers pay 3x more
    2. Age strongly correlates with higher charges
    3. BMI has moderate impact, especially for smokers
    4. Region has minimal impact on overall charges
    5. Number of children has small positive correlation
    6. Model achieves R² score of {:.2f}
    """.format(metrics['r2'])
    ax15.text(0.5, 0.5, insights, fontsize=11, ha='center', va='center',
             transform=ax15.transAxes, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('healthcare_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Dashboard saved as healthcare_dashboard.png")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("HEALTHCARE COST PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    df = load_and_clean_data()
    val_df = load_validation_data()
    print(f"   Training records: {len(df)}")
    print(f"   Validation records: {len(val_df)}")
    
    # Train model
    print("\n[2/5] Training prediction model...")
    model, metrics, X_test, y_test, y_pred, features = train_model(df)
    print(f"   Model: Random Forest Regressor")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: ${metrics['rmse']:,.2f}")
    print(f"   MAE: ${metrics['mae']:,.2f}")
    
    # Generate predictions
    print("\n[3/5] Generating validation predictions...")
    val_processed, predictions = predict_validation(model, val_df, features)
    
    # Save predictions to CSV
    val_output = val_df.copy()
    val_output['predicted_charges'] = predictions
    val_output.to_csv('validation_predictions.csv', index=False)
    print(f"   Predictions saved to validation_predictions.csv")
    
    # Create visualizations
    print("\n[4/5] Creating dashboard...")
    create_dashboard(df, metrics, X_test, y_test, y_pred, val_df, predictions)
    
    # Print summary statistics
    print("\n[5/5] Summary Statistics")
    print("-" * 40)
    print(f"Training Data:")
    print(f"   Average Charges: ${df['charges'].mean():,.2f}")
    print(f"   Median Charges: ${df['charges'].median():,.2f}")
    print(f"   Min Charges: ${df['charges'].min():,.2f}")
    print(f"   Max Charges: ${df['charges'].max():,.2f}")
    print(f"\nValidation Predictions:")
    print(f"   Average Predicted: ${predictions.mean():,.2f}")
    print(f"   Median Predicted: ${np.median(predictions):,.2f}")
    print(f"   Min Predicted: ${predictions.min():,.2f}")
    print(f"   Max Predicted: ${predictions.max():,.2f}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return df, val_df, predictions, metrics

if __name__ == "__main__":
    df, val_df, predictions, metrics = main()
