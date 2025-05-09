import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class DataQualityMonitor:
    """A class for monitoring and improving data quality using AI techniques."""
    
    def __init__(self, contamination=0.01, n_neighbors=5):
        """
        Initialize the DataQualityMonitor.
        
        Parameters:
        - contamination: Expected proportion of anomalies in the dataset
        - n_neighbors: Number of neighbors to use for KNN imputation
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load data from a CSV file."""
        self.df_original = pd.read_csv(file_path)
        self.df = self.df_original.copy()
        print(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def initial_assessment(self):
        """Perform initial data quality assessment."""
        print("\n=== Initial Data Sample ===")
        print(self.df.head())
        
        print("\n=== Data Types ===")
        print(self.df.dtypes)
        
        print("\n=== Summary Statistics ===")
        print(self.df.describe(include='all'))
        
        print("\n=== Missing Values ===")
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })
        print(missing_info)
        
        # Visualize missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Visualization')
        plt.tight_layout()
        plt.savefig('missing_values.png')
        plt.close()
        
        print("\n=== Duplicate Entries ===")
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate entries: {duplicates} ({duplicates/len(self.df)*100:.2f}%)")
        
        return missing_info
    
    def preprocess_data(self):
        """Apply basic preprocessing to prepare data for analysis."""
        print("\n=== Data Preprocessing ===")
        
        # Store the original shape for reporting
        original_shape = self.df.shape
        
        # Convert date columns
        if 'invoice_date' in self.df.columns:
            self.df['invoice_date'] = pd.to_datetime(self.df['invoice_date'], errors='coerce')
            invalid_dates = self.df['invoice_date'].isna().sum()
            print(f"- Converted invoice_date to datetime (found {invalid_dates} invalid dates)")
        
        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace=True)
        print(f"- Removed {duplicates} duplicate entries")
        
        # Basic handling of missing categorical values
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            missing = self.df[col].isna().sum()
            if missing > 0:
                self.df[col].fillna('Unknown', inplace=True)
                print(f"- Temporarily filled {missing} missing values in '{col}' with 'Unknown'")
        
        # Basic handling of missing numerical values (will be refined by KNN later)
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            missing = self.df[col].isna().sum()
            if missing > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                print(f"- Temporarily filled {missing} missing values in '{col}' with median")
        
        # Reset index
        self.df.reset_index(drop=True, inplace=True)
        
        # Report changes
        current_shape = self.df.shape
        print(f"\nPreprocessing complete: {original_shape[0]} rows → {current_shape[0]} rows")
        
        return self.df
    
    def detect_anomalies(self, numeric_features=None):
        """
        Detect anomalies in numerical features using Isolation Forest.
        
        Parameters:
        - numeric_features: List of numerical column names to use for anomaly detection
        """
        if numeric_features is None:
            numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\n=== Anomaly Detection on {len(numeric_features)} features ===")
        print(f"Features used: {numeric_features}")
        
        # Ensure no missing values for anomaly detection
        X = self.df[numeric_features].copy()
        
        # Apply Isolation Forest
        self.df['anomaly'] = self.iso_forest.fit_predict(X)
        self.df['anomaly_score'] = self.iso_forest.decision_function(X)
        
        # Convert predictions to binary labels (1: normal, 0: anomaly)
        self.df['anomaly'] = np.where(self.df['anomaly'] == 1, 0, 1)  # Convert to 0/1 for clarity
        
        anomalies = self.df[self.df['anomaly'] == 1]
        print(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(self.df)*100:.2f}% of data)")
        
        return anomalies
    
    def apply_knn_imputation(self, numeric_features=None):
        """
        Apply KNN imputation to refine the handling of missing values.
        
        Parameters:
        - numeric_features: List of numerical column names to impute
        """
        if numeric_features is None:
            numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove anomaly columns if they exist
            numeric_features = [col for col in numeric_features if col not in ['anomaly', 'anomaly_score']]
        
        print(f"\n=== KNN Imputation on {len(numeric_features)} features ===")
        print(f"Features imputed: {numeric_features}")
        
        # Store original values for comparison
        original_values = self.df[numeric_features].copy()
        
        # Apply KNN imputation
        self.df[numeric_features] = self.imputer.fit_transform(self.df[numeric_features])
        
        # Count number of imputed values
        num_imputed = ((original_values.isna()) & (~self.df[numeric_features].isna())).sum().sum()
        print(f"Refined {num_imputed} previously missing values using KNN imputation")
        
        return self.df[numeric_features]
    
    def standardize_features(self, numeric_features=None):
        """
        Standardize numerical features to have zero mean and unit variance.
        
        Parameters:
        - numeric_features: List of numerical column names to standardize
        """
        if numeric_features is None:
            numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove anomaly columns if they exist
            numeric_features = [col for col in numeric_features if col not in ['anomaly', 'anomaly_score']]
        
        print(f"\n=== Feature Standardization ===")
        print(f"Standardizing {len(numeric_features)} features")
        
        # Create a new DataFrame for standardized features to preserve originals
        self.df_standardized = self.df.copy()
        self.df_standardized[numeric_features] = self.scaler.fit_transform(self.df[numeric_features])
        
        print("Features standardized to zero mean and unit variance")
        
        return self.df_standardized[numeric_features]
    
    def visualize_anomalies(self, x_feature, y_feature, figsize=(10, 6)):
        """
        Visualize detected anomalies in a scatter plot.
        
        Parameters:
        - x_feature: Feature for x-axis
        - y_feature: Feature for y-axis
        - figsize: Size of the figure
        """
        if 'anomaly' not in self.df.columns:
            print("Error: Run detect_anomalies() before visualization")
            return
        
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        sns.scatterplot(
            data=self.df,
            x=x_feature,
            y=y_feature,
            hue='anomaly',
            palette={0: 'blue', 1: 'red'},
            alpha=0.7
        )
        
        plt.title(f'Anomaly Detection: {y_feature} vs {x_feature}')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.legend(title='Anomaly', labels=['Normal', 'Anomaly'])
        plt.tight_layout()
        plt.savefig('anomaly_detection.png')
        plt.close()
        
        print(f"Anomaly visualization saved as 'anomaly_detection.png'")
    
    def generate_report(self):
        """Generate a comprehensive data quality report."""
        print("\n=== Data Quality Report ===")
        
        # Basic statistics
        print(f"Total records processed: {len(self.df)}")
        
        # Missing value summary after processing
        missing_after = self.df.isnull().sum()
        print("\nRemaining missing values:")
        print(missing_after[missing_after > 0])
        
        # Anomaly summary
        if 'anomaly' in self.df.columns:
            anomalies = self.df[self.df['anomaly'] == 1]
            print(f"\nAnomalies detected: {len(anomalies)} ({len(anomalies)/len(self.df)*100:.2f}%)")
            
            # Summary of anomalies by category (if applicable)
            if 'category' in self.df.columns:
                print("\nAnomalies by category:")
                anomaly_by_category = self.df.groupby('category')['anomaly'].sum()
                category_counts = self.df.groupby('category').size()
                anomaly_percentage = (anomaly_by_category / category_counts) * 100
                anomaly_summary = pd.DataFrame({
                    'Total Records': category_counts,
                    'Anomalies': anomaly_by_category,
                    'Percentage': anomaly_percentage
                }).sort_values('Percentage', ascending=False)
                print(anomaly_summary)
        
        # Generate timestamp for the report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nReport generated: {timestamp}")
        
        return {
            'total_records': len(self.df),
            'missing_values': missing_after.to_dict(),
            'anomalies': len(anomalies) if 'anomaly' in self.df.columns else 0,
            'timestamp': timestamp
        }


# Example of usage
if __name__ == "__main__":
    # Initialize the monitor
    monitor = DataQualityMonitor(contamination=0.01, n_neighbors=5)
    
    # Load data
    df = monitor.load_data('customer_shopping_data.csv')
    
    # Initial assessment
    monitor.initial_assessment()
    
    # Preprocess data
    monitor.preprocess_data()
    
    # Feature selection
    numeric_features = ['age', 'quantity', 'price']
    
    # Detect anomalies
    anomalies = monitor.detect_anomalies(numeric_features)
    
    # Apply KNN imputation
    monitor.apply_knn_imputation(numeric_features)
    
    # Standardize features
    monitor.standardize_features(numeric_features)
    
    # Visualize anomalies
    monitor.visualize_anomalies('quantity', 'price')
    
    # Generate comprehensive report
    report = monitor.generate_report()
    
    print("\nData quality monitoring complete!")