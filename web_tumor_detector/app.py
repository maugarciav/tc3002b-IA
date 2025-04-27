from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp' 

# Cargar el modelo Keras
try:
    model = tf.keras.models.load_model('') #!!!1NOMBRE DEL MODELO QUE QUIERS PROBAR!!!!!
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

def preparar_imagen(ruta_imagen, img_height=224, img_width=224):
    """Carga y preprocesa la imagen para la predicción."""
    img = image.load_img(ruta_imagen, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  
    return img_array

@app.route('/temp/<filename>')
def mostrar_imagen(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    probabilidades = None
    ruta_imagen_subida = None
    class_labels = {0: 'Glioma tumor', 1: 'Meningioma Tumor', 2: 'Normal', 3: 'Pituitary Tumor'}

    if request.method == 'POST' and model is not None:
        if 'imagen' in request.files:
            imagen = request.files['imagen']
            nombre_seguro = secure_filename(imagen.filename)
            ruta_guardado = os.path.join(app.config['UPLOAD_FOLDER'], nombre_seguro)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            imagen.save(ruta_guardado)
            ruta_imagen_subida = os.path.join('temp', nombre_seguro) # La ruta para el navegador

            try:
                img_preparada = preparar_imagen(ruta_guardado)
                predicciones = model.predict(img_preparada)

                predicted_class_index = np.argmax(predicciones[0])
                predicted_class_name = class_labels.get(predicted_class_index, f"Clase desconocida ({predicted_class_index})")
                confidence = predicciones[0][predicted_class_index] * 100
                prediccion = f" {predicted_class_name} (Confianza: {confidence:.2f}%)"

                probabilidades = {}
                for i, prob in enumerate(predicciones[0]):
                    class_name = class_labels.get(i, f"Clase {i}")
                    probabilidades[class_name] = prob * 100

            except Exception as e:
                prediccion = f"Error al procesar la imagen: {e}"

    return render_template('index.html', prediccion=prediccion, probabilidades=probabilidades, ruta_imagen_subida=ruta_imagen_subida)

if __name__ == '__main__':
    app.run(debug=True)