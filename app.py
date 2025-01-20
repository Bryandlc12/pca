from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No se subió ningún archivo", 400
    
    file = request.files['file']
    if file.filename == '':
        return "El archivo está vacío", 400
    
    # Guardar el archivo subido
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Leer el archivo como DataFrame
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error al leer el archivo: {e}", 400

    # Verificar si hay suficientes columnas
    if df.shape[1] < 2:
        return "El archivo debe tener al menos dos columnas para realizar PCA", 400

    # Filtrar solo columnas numéricas
    df_numeric = df.select_dtypes(include=['number'])

    # Verificar que haya suficientes columnas numéricas
    if df_numeric.shape[1] < 2:
        return "El archivo debe tener al menos dos columnas numéricas para realizar PCA", 400

    # Estandarizar los datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # Aplicar PCA
    pca = PCA(n_components=min(2, df_numeric.shape[1]))
    principal_components = pca.fit_transform(scaled_data)

    # Crear un nuevo DataFrame con los componentes principales
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

    # Guardar los resultados en un archivo CSV
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pca_result.csv')
    pca_df.to_csv(output_path, index=False)

    return redirect(url_for('download_file', filename='pca_result.csv'))


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
