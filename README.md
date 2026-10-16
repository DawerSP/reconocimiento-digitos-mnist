# Clasificación de Dígitos Manuscritos mediante Redes Neuronales Convolucionales y Preprocesamiento Digital de Imágenes

## Resumen del Proyecto

Este proyecto presenta la implementación de un sistema de visión por computadora y aprendizaje profundo diseñado para la clasificación de dígitos manuscritos (del 0 al 9) en tiempo real. La arquitectura utiliza una Red Neuronal Convolucional (CNN) entrenada sobre el conjunto de datos estandarizado **MNIST** (*Modified National Institute of Standards and Technology*). 

Para garantizar la robustez ante entradas del usuario con diferentes grosores, proporciones y ubicaciones, se integró un pipeline de preprocesamiento digital de imágenes mediante **OpenCV** previo a la inferencia del modelo. La interfaz gráfica interactiva fue desarrollada con la librería **Gradio**.

## Pipeline de Preprocesamiento de Imágenes

Para adaptar la entrada dibujada por el usuario al formato específico con el que fue entrenada la red neuronal, la imagen pasa por los siguientes procesos secuenciales:

1. **Inversión de Espectro de Color (`cv2.bitwise_not`):** Transforma la imagen de trazo negro sobre lienzo blanco a un formato de trazo blanco sobre fondo negro.
2. **Filtrado de Dilatación Morfológica (`cv2.dilate`):** Aplica una operación de dilatación con un kernel estructurado de $3 \times 3$ píxeles. Esto refuerza los trazos delgados (evitando la mala clasificación de números como el 1 o el 7) sin saturar las cavidades internas de números complejos como el 6, el 8 o el 0.
3. **Detección de Contorno y Segmentación (`cv2.boundingRect`):** Identifica el área delimitadora del dígito dibujado y extrae la región de interés (ROI).
4. **Redimensionamiento Proporcional y Centrado Estandarizado:** 
   * Redimensiona la región extraída preservando la relación de aspecto dentro de un marco interno de $20 \times 20$ píxeles.
   * Modula la imagen dentro de una matriz nula de $28 \times 28$ píxeles, posicionando el centro de masa del número en el centro geométrico del lienzo.
5. **Normalización de Valores:** Escala la matriz de intensidad de píxeles del rango discreto $[0, 255]$ al rango continuo $[0.0, 1.0]$.

## Especificaciones Técnicas y Entorno

* **Lenguaje de Programación:** Python 3.10+ (compatible con Python 3.13)
* **Marco de Aprendizaje Profundo:** TensorFlow / Keras
* **Procesamiento de Imágenes:** OpenCV (`opencv-python`)
* **Cálculo Numérico:** NumPy
* **Despliegue de Interfaz:** Gradio

## Requisitos e Instalación

### 1. Clonación del Repositorio

```bash
git clone https://github.com/DawerSP/reconocimiento-digitos-mnist.git
cd reconocimiento-digitos-mnist
```

### 2. Instalación de Dependencias

Se recomienda la ejecución dentro de un entorno virtual de Python. Las dependencias se pueden instalar mediante el gestor de paquetes `pip`:

```bash
pip install tensorflow opencv-python numpy gradio
```

## Estructura del Código Fuente y Inferencia

Asegúrese de incluir el archivo con los pesos del modelo entrenado (`numerosC2.keras`) en la raíz del directorio del proyecto.

```python
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Carga del modelo entrenado
modelo = tf.keras.models.load_model('numerosC2.keras')

def preprocesar_y_predecir(imagen):
    if imagen is None:
        return {}
    
    # Extracción del canal de imagen desde la interfaz
    if isinstance(imagen, dict):
        imagen = imagen.get("composite", imagen.get("background", list(imagen.values())[0]))
    
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
        
    # Inversión de imagen
    imagen = cv2.bitwise_not(imagen)
    
    # Aplicación de dilatación suave
    kernel = np.ones((3, 3), np.uint8)
    imagen = cv2.dilate(imagen, kernel, iterations=1)
    
    # Recorte y centrado dinámico del dígito
    coords = cv2.findNonZero(imagen)
    if coords is None:
        return {str(i): 0.0 for i in range(10)}
    
    x, y, w, h = cv2.boundingRect(coords)
    recorte = imagen[y:y+h, x:x+w]
    
    if w > h:
        new_w = 20
        new_h = int(h * (20.0 / w))
    else:
        new_h = 20
        new_w = int(w * (20.0 / h))
    
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    
    recorte_resized = cv2.resize(recorte, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Matriz de 28x28 con centrado automático
    canvas_28x28 = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas_28x28[start_y:start_y+new_h, start_x:start_x+new_w] = recorte_resized
    
    # Normalización del vector de entrada
    imagen_normalized = canvas_28x28.astype(np.float32) / 255.0
    imagen_input = np.expand_dims(imagen_normalized, axis=(-1, 0))
    
    # Inferencia del modelo
    prediccion = modelo.predict(imagen_input, verbose=0)[0]
    return {str(i): float(prediccion[i]) for i in range(10)}

# Despliegue de la interfaz interactiva
demo = gr.Interface(
    fn=preprocesar_y_predecir,
    inputs=gr.Sketchpad(type="numpy", label="Lienzo de dibujo para dígito manuscrito"),
    outputs=gr.Label(num_top_classes=3, label="Distribución de Probabilidad"),
    title="Sistema de Reconocimiento de Dígitos Manuscritos (MNIST - CNN)"
)

demo.launch()
```

## Autoría

* **Desarrollador:** DawerSP
* **Proyecto Académico:** Reconocimiento de Patrones y Visión por Computadora