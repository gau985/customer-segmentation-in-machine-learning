import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# ML Model Functions
def apply_kmeans(df):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    kmeans = KMeans(n_clusters=2, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X)
    return df, "KMeans Clustering"

def apply_hierarchical(df):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    model = AgglomerativeClustering(n_clusters=2)
    df['Cluster'] = model.fit_predict(X)
    return df, "Hierarchical Clustering"

def apply_gmm(df):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    gmm = GaussianMixture(n_components=2, random_state=0)
    df['Cluster'] = gmm.fit_predict(X)
    return df, "Gaussian Mixture Model"

def apply_knn(df):
    if 'Gender' not in df.columns:
        return df, "KNN (Missing 'Gender' column)"
    
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df.dropna(subset=['Gender'], inplace=True)

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    y = df['Gender']
    knn = KNeighborsClassifier(n_neighbors=3)
    df['Predicted Gender'] = knn.fit(X, y).predict(X)
    return df, "KNN Classification"

def apply_regression(df):
    X = df[['Annual Income (k$)']]
    y = df['Spending Score (1-100)']
    model = LinearRegression()
    df['Predicted Spending Score'] = model.fit(X, y).predict(X)
    return df, "Linear Regression"

@app.route('/', methods=['GET', 'POST'])
def index():
    scatter_plot = bar_chart = corr_matrix = pie_chart = None

    if request.method == 'POST':
        file = request.files['dataset']
        model = request.form['model']

        if file:
            df = pd.read_csv(file)
            df.dropna(inplace=True)
            df = df.head(100).sort_values(by='Annual Income (k$)', ascending=True)

            # Apply selected model
            if model == 'kmeans':
                df, result = apply_kmeans(df)
            elif model == 'hierarchical':
                df, result = apply_hierarchical(df)
            elif model == 'gmm':
                df, result = apply_gmm(df)
            elif model == 'knn':
                df, result = apply_knn(df)
            elif model == 'regression':
                df, result = apply_regression(df)

            # Plot directory
            static_dir = os.path.join(app.root_path, 'static')
            os.makedirs(static_dir, exist_ok=True)

            # Scatter Plot
            scatter_plot = 'scatter_plot.png'
            plt.figure(figsize=(6, 4))
            if 'Cluster' in df:
                sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df)
            elif 'Predicted Gender' in df:
                sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Predicted Gender', data=df)
            elif 'Predicted Spending Score' in df:
                plt.plot(df['Annual Income (k$)'], df['Spending Score (1-100)'], 'o', label='Actual')
                plt.plot(df['Annual Income (k$)'], df['Predicted Spending Score'], '-', label='Predicted')
                plt.legend()
            else:
                sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
            plt.title("Scatter Plot")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, scatter_plot))
            plt.close()

            # Bar Chart
            bar_chart = 'bar_chart.png'
            plt.figure(figsize=(6, 4))
            df.groupby('Annual Income (k$)')['Spending Score (1-100)'].mean().plot(kind='bar', color='skyblue')
            plt.title("Bar Chart: Avg Spending Score by Income")
            plt.ylabel("Avg Spending Score")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, bar_chart))
            plt.close()

            # Correlation Matrix
            corr_matrix = 'correlation_matrix.png'
            corr = df[['Annual Income (k$)', 'Spending Score (1-100)']].corr()
            plt.figure(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, corr_matrix))
            plt.close()

            # Pie Chart
            pie_chart = 'pie_chart.png'
            plt.figure(figsize=(6, 4))
            if 'Cluster' in df:
                df['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set2'))
                plt.title(f"{result} - Cluster Pie")
            elif 'Predicted Gender' in df:
                df['Predicted Gender'].replace({0: 'Male', 1: 'Female'}).value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Set1'))
                plt.title(f"{result} - Gender Pie")
            else:
                df['Spending Score (1-100)'].value_counts(bins=5).plot(kind='pie', autopct='%1.1f%%')
                plt.title(f"{result} - Score Distribution")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, pie_chart))
            plt.close()

    return render_template("index.html",
                           scatter_plot=scatter_plot,
                           bar_chart=bar_chart,
                           corr_matrix=corr_matrix,
                           pie_chart=pie_chart)

if __name__ == '__main__':
    app.run(debug=True)
