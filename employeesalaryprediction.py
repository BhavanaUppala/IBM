# Employee Salary Prediction System - Complete Implementation
# Capstone Project for IBM SkillsBuild AI Internship

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA GENERATION ====================

def generate_sample_data():
    """Generate comprehensive sample employee salary data"""
    np.random.seed(42)
    
    # Define data parameters
    n_samples = 1000
    
    # Job titles and their base salary ranges
    job_titles = {
        'Software Engineer': (60000, 120000),
        'Data Scientist': (70000, 140000),
        'Product Manager': (75000, 130000),
        'Marketing Manager': (55000, 100000),
        'Sales Representative': (40000, 80000),
        'HR Manager': (50000, 95000),
        'Financial Analyst': (55000, 105000),
        'Operations Manager': (60000, 110000),
        'UX Designer': (55000, 105000),
        'Business Analyst': (50000, 95000)
    }
    
    # Education levels and their multipliers
    education_levels = {
        'High School': 0.8,
        'Bachelor': 1.0,
        'Master': 1.25,
        'PhD': 1.5
    }
    
    # Locations and their cost of living multipliers
    locations = {
        'New York': 1.3,
        'California': 1.25,
        'Texas': 1.0,
        'Florida': 0.95,
        'Illinois': 1.1,
        'Washington': 1.2,
        'Massachusetts': 1.15,
        'Colorado': 1.05,
        'Georgia': 0.9,
        'North Carolina': 0.85
    }
    
    # Company sizes and their multipliers
    company_sizes = {
        'Startup': 0.9,
        'Small': 0.95,
        'Medium': 1.0,
        'Large': 1.15,
        'Enterprise': 1.25
    }
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Random selections
        job_title = np.random.choice(list(job_titles.keys()))
        education = np.random.choice(list(education_levels.keys()))
        location = np.random.choice(list(locations.keys()))
        company_size = np.random.choice(list(company_sizes.keys()))
        
        # Generate experience (0-20 years)
        experience = np.random.randint(0, 21)
        
        # Calculate base salary
        base_min, base_max = job_titles[job_title]
        base_salary = np.random.uniform(base_min, base_max)
        
        # Apply multipliers
        salary = base_salary * education_levels[education] * locations[location] * company_sizes[company_size]
        
        # Add experience bonus (2-5% per year)
        experience_bonus = 1 + (experience * np.random.uniform(0.02, 0.05))
        salary *= experience_bonus
        
        # Add some random variation (±10%)
        salary *= np.random.uniform(0.9, 1.1)
        
        # Round to nearest 100
        salary = round(salary, -2)
        
        data.append({
            'Job_Title': job_title,
            'Years_Experience': experience,
            'Education_Level': education,
            'Location': location,
            'Company_Size': company_size,
            'Salary': salary
        })
    
    return pd.DataFrame(data)

# ==================== DATA PREPROCESSING ====================

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def fit_transform(self, df):
        """Fit and transform the data"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Job_Title', 'Education_Level', 'Location', 'Company_Size']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Select features for training
        self.feature_columns = ['Years_Experience'] + [col + '_Encoded' for col in categorical_columns]
        X = df_processed[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df_processed['Salary']
    
    def transform(self, df):
        """Transform new data using fitted encoders and scaler"""
        df_processed = df.copy()
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            df_processed[col + '_Encoded'] = le.transform(df_processed[col])
        
        # Select and scale features
        X = df_processed[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

# ==================== MODEL TRAINING ====================

class SalaryPredictor:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.preprocessor = DataPreprocessor()
    
    def train_models(self, df):
        """Train and evaluate all models"""
        print("Training Models...")
        print("="*50)
        
        # Preprocess data
        X, y = self.preprocessor.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_r2 = -float('inf')
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Store results
            self.results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name}:")
            print(f"  R² Score: {r2:.3f}")
            print(f"  MAE: ${mae:,.0f}")
            print(f"  RMSE: ${rmse:,.0f}")
            print(f"  CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print("-" * 30)
            
            # Track best model
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"Best Model: {self.best_model_name} with R² = {best_r2:.3f}")
        return self.results
    
    def predict_salary(self, job_title, experience, education, location, company_size):
        """Predict salary for given parameters"""
        if self.best_model is None:
            return "Model not trained yet"
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Job_Title': [job_title],
            'Years_Experience': [experience],
            'Education_Level': [education],
            'Location': [location],
            'Company_Size': [company_size]
        })
        
        # Preprocess input
        X_input = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.best_model.predict(X_input)[0]
        
        # Calculate prediction interval (rough estimate)
        rmse = self.results[self.best_model_name]['rmse']
        lower_bound = prediction - 1.96 * rmse
        upper_bound = prediction + 1.96 * rmse
        
        return prediction, lower_bound, upper_bound

# ==================== VISUALIZATION ====================

def create_visualizations(df, results):
    """Create comprehensive visualizations"""
    
    # Set style
    plt.style.use('default')
    
    # 1. Salary Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['Salary'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Salaries')
    plt.xlabel('Salary ($)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 2. Salary by Experience
    plt.subplot(2, 2, 2)
    experience_groups = df.groupby('Years_Experience')['Salary'].mean()
    plt.plot(experience_groups.index, experience_groups.values, marker='o', linewidth=2, markersize=6)
    plt.title('Average Salary by Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Average Salary ($)')
    plt.grid(True, alpha=0.3)
    
    # 3. Salary by Job Title
    plt.subplot(2, 2, 3)
    job_salary = df.groupby('Job_Title')['Salary'].mean().sort_values(ascending=True)
    plt.barh(range(len(job_salary)), job_salary.values, color='lightgreen')
    plt.yticks(range(len(job_salary)), job_salary.index)
    plt.title('Average Salary by Job Title')
    plt.xlabel('Average Salary ($)')
    
    # 4. Salary by Education Level
    plt.subplot(2, 2, 4)
    edu_order = ['High School', 'Bachelor', 'Master', 'PhD']
    edu_salary = df.groupby('Education_Level')['Salary'].mean().reindex(edu_order)
    bars = plt.bar(edu_salary.index, edu_salary.values, color='lightcoral')
    plt.title('Average Salary by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Average Salary ($)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, edu_salary.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'${value:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Model Performance Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    
    bars = plt.bar(model_names, r2_scores, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Performance Comparison (R² Score)')
    plt.ylabel('R² Score')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    
    # 6. Prediction vs Actual (Best Model)
    plt.subplot(1, 2, 2)
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    y_test = results[best_model_name]['y_test']
    y_pred = results[best_model_name]['y_pred']
    
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Salary ($)')
    plt.ylabel('Predicted Salary ($)')
    plt.title(f'Actual vs Predicted Salary ({best_model_name})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 7. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    
    # Create numeric dataframe for correlation
    df_numeric = df.copy()
    le = LabelEncoder()
    categorical_cols = ['Job_Title', 'Education_Level', 'Location', 'Company_Size']
    
    for col in categorical_cols:
        df_numeric[col + '_Numeric'] = le.fit_transform(df_numeric[col])
    
    # Select numeric columns for correlation
    numeric_cols = ['Years_Experience', 'Salary'] + [col + '_Numeric' for col in categorical_cols]
    correlation_matrix = df_numeric[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# ==================== STREAMLIT WEB APP ====================

def create_streamlit_app():
    """Create Streamlit web application"""
    
    st.title("💼 Employee Salary Prediction System")
    st.write("Predict employee salaries using AI and machine learning")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
        st.session_state.df = None
    
    # Sidebar for model training
    st.sidebar.header("🚀 Model Training")
    if st.sidebar.button("Train Model"):
        with st.spinner("Training models... This may take a moment."):
            # Generate data and train model
            df = generate_sample_data()
            predictor = SalaryPredictor()
            results = predictor.train_models(df)
            
            # Store in session state
            st.session_state.predictor = predictor
            st.session_state.df = df
            st.session_state.results = results
            
            st.sidebar.success("✅ Model trained successfully!")
    
    # Main prediction interface
    if st.session_state.predictor is not None:
        st.header("🎯 Salary Prediction")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.selectbox("Job Title", [
                'Software Engineer', 'Data Scientist', 'Product Manager',
                'Marketing Manager', 'Sales Representative', 'HR Manager',
                'Financial Analyst', 'Operations Manager', 'UX Designer', 'Business Analyst'
            ])
            
            experience = st.slider("Years of Experience", 0, 20, 5)
            
            education = st.selectbox("Education Level", [
                'High School', 'Bachelor', 'Master', 'PhD'
            ])
        
        with col2:
            location = st.selectbox("Location", [
                'New York', 'California', 'Texas', 'Florida', 'Illinois',
                'Washington', 'Massachusetts', 'Colorado', 'Georgia', 'North Carolina'
            ])
            
            company_size = st.selectbox("Company Size", [
                'Startup', 'Small', 'Medium', 'Large', 'Enterprise'
            ])
        
        # Predict button
        if st.button("🔮 Predict Salary", type="primary"):
            try:
                prediction, lower_bound, upper_bound = st.session_state.predictor.predict_salary(
                    job_title, experience, education, location, company_size
                )
                
                # Display results
                st.success("✅ Prediction completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💰 Predicted Salary", f"${prediction:,.0f}")
                
                with col2:
                    st.metric("📉 Lower Bound", f"${lower_bound:,.0f}")
                
                with col3:
                    st.metric("📈 Upper Bound", f"${upper_bound:,.0f}")
                
                # Confidence interval
                st.info(f"🎯 **Prediction Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
        
        # Model performance section
        st.header("📊 Model Performance")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Performance metrics table
            performance_data = []
            for name, result in results.items():
                performance_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2']:.3f}",
                    'MAE': f"${result['mae']:,.0f}",
                    'RMSE': f"${result['rmse']:,.0f}",
                    'Status': '🏆' if name == st.session_state.predictor.best_model_name else ''
                })
            
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)
            
            # Best model info
            best_model_name = st.session_state.predictor.best_model_name
            best_r2 = results[best_model_name]['r2']
            st.success(f"🏆 **Best Model:** {best_model_name} with R² Score of {best_r2:.3f}")
        
        # Dataset insights
        st.header("📈 Dataset Insights")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Records", len(df))
            
            with col2:
                st.metric("💰 Avg Salary", f"${df['Salary'].mean():,.0f}")
            
            with col3:
                st.metric("🔝 Max Salary", f"${df['Salary'].max():,.0f}")
            
            with col4:
                st.metric("📊 Job Titles", df['Job_Title'].nunique())
            
            # Show sample data
            with st.expander("📋 View Sample Data"):
                st.dataframe(df.head(10))
    
    else:
        st.info("👆 Please train the model using the sidebar to start making predictions!")
        
        # Show sample data structure
        st.header("📋 Sample Data Structure")
        sample_df = generate_sample_data().head(5)
        st.dataframe(sample_df)

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run the complete pipeline"""
    print("Employee Salary Prediction System")
    print("="*50)
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    df = generate_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Salary statistics:")
    print(df['Salary'].describe())
    
    # Step 2: Initialize and train predictor
    print("\nStep 2: Training models...")
    predictor = SalaryPredictor()
    results = predictor.train_models(df)
    
    # Step 3: Create visualizations
    print("\nStep 3: Creating visualizations...")
    create_visualizations(df, results)
    
    # Step 4: Test predictions
    print("\nStep 4: Testing sample predictions...")
    test_cases = [
        ('Software Engineer', 5, 'Bachelor', 'New York', 'Medium'),
        ('Data Scientist', 8, 'Master', 'California', 'Large'),
        ('Marketing Manager', 3, 'Bachelor', 'Texas', 'Small'),
        ('Product Manager', 12, 'Master', 'Washington', 'Enterprise')
    ]
    
    for job_title, experience, education, location, company_size in test_cases:
        prediction, lower, upper = predictor.predict_salary(
            job_title, experience, education, location, company_size
        )
        print(f"\nPrediction for {job_title}:")
        print(f"  Experience: {experience} years")
        print(f"  Education: {education}")
        print(f"  Location: {location}")
        print(f"  Company: {company_size}")
        print(f"  Predicted Salary: ${prediction:,.0f}")
        print(f"  Range: ${lower:,.0f} - ${upper:,.0f}")
        print("-" * 50)
    
    print("\nProject completed successfully!")
    print("To run the web app, use: streamlit run this_file.py")

# Save and load model functions
def save_model(predictor, filename='salary_predictor.pkl'):
    """Save trained model"""
    joblib.dump(predictor, filename)
    print(f"Model saved to {filename}")

def load_model(filename='salary_predictor.pkl'):
    """Load trained model"""
    predictor = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return predictor

# Run the application
if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        # This will work if running in Streamlit
        create_streamlit_app()
    except:
        main()