import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
# Asegúrate de que la carpeta uploads esté dentro del directorio del proyecto
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Verifica si la carpeta 'uploads' existe, si no, la crea
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])



# Asegurarse de que la carpeta de subida exista
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No se subió ningún archivo", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "El archivo está vacío", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Leer archivo CSV
        df = pd.read_csv(filepath)
        
        # Preprocesamiento de datos
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))  # Solo columnas numéricas
        
        # PCA
        pca = PCA()
        principal_components = pca.fit_transform(scaled_data)

        # Graficar Scree Plot
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Componentes principales')
        plt.ylabel('Varianza explicada')
        scree_plot_path = 'static/images/scree_plot.png'
        plt.savefig(scree_plot_path)
        plt.close()

        # Graficar PCA Scatter Plot (PC1 vs PC2)
        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolors='k')
        plt.title('PCA: Componentes Principales')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        pca_scatter_path = 'static/images/pca_scatter.png'
        plt.savefig(pca_scatter_path)
        plt.close()

        # Graficar Loading Plot
        loadings = pca.components_.T
        plt.figure(figsize=(8, 6))
        plt.scatter(loadings[:, 0], loadings[:, 1], edgecolors='r')
        plt.title('Loading Plot')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        loading_plot_path = 'static/images/loading_plot.png'
        plt.savefig(loading_plot_path)
        plt.close()

        # Graficar Biplot
        plt.figure(figsize=(8, 6))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolors='k')
        for i in range(loadings.shape[0]):
            plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
            plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, df.columns[i], color='r', ha='center', va='center')
        plt.title('Biplot de PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        biplot_path = 'static/images/biplot.png'
        plt.savefig(biplot_path)
        plt.close()

        return render_template('index.html', 
                               scree_plot=scree_plot_path, 
                               pca_scatter=pca_scatter_path, 
                               loading_plot=loading_plot_path, 
                               biplot=biplot_path)

    return "El archivo no es válido", 400

if __name__ == "__main__":
    app.run(debug=True)
