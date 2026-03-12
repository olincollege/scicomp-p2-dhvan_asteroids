from algorithms.algorithm import Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from decorators import timer
import plotly.express as px
import numpy as np

class AsteroidKMeans(Algorithm):
    def __init__(self, reload_raw_data = False, algorithm_name = "KMeans", debug_prints = False):
        self.kmeans_model = KMeans(50)
        self.scaler = StandardScaler()
        super().__init__(reload_raw_data, algorithm_name, debug_prints)
        
    @timer
    def fit_predict(self):
        X_scaled = self.scaler.fit_transform(self.X)
        predictions = self.kmeans_model.fit_predict(X_scaled)
        print("\nCluster Breakdown:", np.unique(predictions, return_counts=True))
        self.cached_predictions = predictions
        
        return predictions
    
    # AI Generated visulization code
    def visualize_clusters(self, sample_size=10000):
        # Make sure we have predictions
        predictions = self.fit_predict() if self.cached_predictions is None else self.cached_predictions
        
        # Create a copy of the features so we can add our labels without breaking the model
        plot_df = self.X.copy()
        plot_df['KMeans_Cluster'] = predictions.astype(str) # Convert to string for discrete colors
        plot_df['True_Family'] = self.Y.values
        
        # Sample the data to keep the 3D render smooth
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(n=sample_size, random_state=42)
        
        # Create the interactive 3D scatter plot
        fig = px.scatter_3d(
            plot_df, 
            x='a', 
            y='ecc', 
            z='sinI',
            color='KMeans_Cluster',      # Color the dots by what K-Means guessed
            hover_data=['True_Family'],  # Show the real answer when you hover your mouse
            title=f"3D Cluster Visualization: {self.algorithm_name}",
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Alphabet # Gives a lot of distinct colors
        )
        
        # Make the dots smaller for better visibility of dense areas
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        
        # This will open a new tab in your web browser with the interactive plot
        fig.show()