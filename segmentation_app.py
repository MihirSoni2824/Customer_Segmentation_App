import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Ensure folders exist
os.makedirs("plots", exist_ok=True)

# Load or create data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file.name)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            "CustomerID": range(1, 201),
            "Age": np.random.randint(18, 70, 200),
            "Annual Income": np.random.randint(20000, 150000, 200),
            "Spending Score": np.random.randint(1, 101, 200)
        })
    return df

# Scale features
def scale_data(df, scaler_name):
    features = ["Age", "Annual Income", "Spending Score"]
    X = df[features].values
    scaler = StandardScaler() if scaler_name == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Plot Elbow and Silhouette
def plot_elbow_silhouette(X):
    wcss = []
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure()
    plt.plot(K, wcss, marker='o')
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.savefig("plots/elbow_plot.png")
    plt.close()

    plt.figure()
    plt.plot(K, silhouette_scores, marker='o', color='orange')
    plt.title("Silhouette Scores")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.savefig("plots/silhouette_plot.png")
    plt.close()

# Run KMeans and return summary
def run_kmeans(df, X, k):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    df["Cluster"] = labels
    summary = df.groupby("Cluster")[["Age", "Annual Income", "Spending Score"]].mean().round(1)
    return df, summary

# Plot PCA scatter
def plot_pca(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.title("PCA Scatter Plot")
    plt.savefig("plots/pca_scatter.png")
    plt.close()

# Plot pairplot
def plot_pairplot(df):
    sns.pairplot(df, vars=["Age", "Annual Income", "Spending Score"], hue="Cluster", palette="tab10")
    plt.savefig("plots/pairplot.png")
    plt.close()

# Plot centroid bar chart
def plot_centroid(summary):
    summary.plot(kind="bar")
    plt.title("Cluster Centroids")
    plt.ylabel("Feature Mean")
    plt.tight_layout()
    plt.savefig("plots/centroid_chart.png")
    plt.close()

# Full pipeline
def segmentation_pipeline(file, scaler_name, k):
    df = load_data(file)
    X = scale_data(df, scaler_name)
    plot_elbow_silhouette(X)
    df_clustered, summary = run_kmeans(df, X, k)
    plot_pca(X, df_clustered["Cluster"])
    plot_pairplot(df_clustered)
    plot_centroid(summary)

    interpretation = []
    for i, row in summary.iterrows():
        interpretation.append(
            f"Cluster {i}: Age ~ {row['Age']}, Income ~ {row['Annual Income']}, Score ~ {row['Spending Score']}"
        )
    return (
        df_clustered.head(10).to_markdown(),
        summary.to_markdown(),
        "\n".join(interpretation),
        "plots/elbow_plot.png",
        "plots/silhouette_plot.png",
        "plots/pca_scatter.png",
        "plots/pairplot.png",
        "plots/centroid_chart.png"
    )

# Background CSS
css = """
body {
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    transition: background-image 1s ease-in-out;
}
"""

# Gradio Interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("## Customer Segmentation with K-Means Clustering")

    # Background Image Switcher (JS)
    gr.HTML("""
    <script>
        let images = [
            '/file/static/bg/bg1.jpg',
            '/file/static/bg/bg2.jpg',
            '/file/static/bg/bg3.jpg'
        ];
        let i = 0;
        setInterval(() => {
            document.body.style.backgroundImage = `url('${images[i]}')`;
            i = (i + 1) % images.length;
        }, 7000);
    </script>
    """)

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        scaler = gr.Radio(["StandardScaler", "MinMaxScaler"], label="Scaler", value="StandardScaler")
        k_value = gr.Slider(2, 10, step=1, label="Number of Clusters (k)", value=3)
        btn = gr.Button("Run Clustering")

    with gr.Row():
        df_head = gr.Textbox(label="Sample DataFrame", lines=10)
        cluster_summary = gr.Textbox(label="Cluster Summary", lines=10)
        interpretations = gr.Textbox(label="Interpretations", lines=5)

    with gr.Row():
        img1 = gr.Image(label="Elbow Plot")
        img2 = gr.Image(label="Silhouette Plot")
        img3 = gr.Image(label="PCA Scatter Plot")
    with gr.Row():
        img4 = gr.Image(label="Pairplot")
        img5 = gr.Image(label="Centroid Chart")

    btn.click(fn=segmentation_pipeline,
              inputs=[file_input, scaler, k_value],
              outputs=[df_head, cluster_summary, interpretations, img1, img2, img3, img4, img5])

demo.launch()
