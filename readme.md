# 📘 **AI Lab – Imágenes, Sonidos y Posturas**  
**IA-VISION-CODERS**

Laboratorio interactivo de Inteligencia Artificial completamente ejecutado **en el navegador** con **TensorFlow.js**, diseñado para estudiantes que aprenden a crear, entrenar y desplegar modelos de visión, audio y posturas.

Este proyecto demuestra cómo integrar **modelos de IA multimodal** en una web responsiva usando únicamente **HTML, CSS y JavaScript puro**, sin frameworks, sin bundlers y totalmente compatible con **GitHub Pages**.

---

## 🌐 **Demo en vivo**  
> *(Cuando lo publiques, reemplaza este link)*  
👉 https://tuusuario.github.io/ai-lab/

---

# 🚀 Características

### 🎥 **1. Modo Imágenes – Accesorios en el rostro**
- Usa la webcam o una imagen cargada.
- Modelo entrenado en **Teachable Machine (Imagen)**.
- Clases:
  - con_hoodie (Con capucha)
  - con_gafas (Con gafas)
  - con_tapabocas (Con tapabocas)
  - sin_accesorios (Ninguno)
- Inferencia en tiempo real, 100% en navegador.

---

### 🎤 **2. Modo Sonidos – Idiomas**
- Grabación desde micrófono o carga de audio `.wav`/`.mp3`.
- Modelo CNN entrenado en **Google Colab con Python**.
- Convertido a **TensorFlow.js** para uso directo en la web.
- Idiomas que reconoce:
  - english  
  - spanish  
  - french  
  - german  
  - russian  

---

### 🧍 **3. Modo Posturas – Acciones tipo deporte/combate**
- Usa webcam o imagen.
- Modelo entrenado en **Teachable Machine (Imagen/Posturas)**.
- Clases:
  - punch  
  - kick  
  - fencing  
  - sword_fight  
  - boxing  

---

# 🏗️ Arquitectura del Proyecto

```
/ (root)
│── index.html           # Interfaz principal
│── styles.css           # Tema oscuro + dashboard responsive
│── app.js               # Lógica JS: cámaras, modelos, audio, UI
│
└── models/
    ├── imagenes/
    │   ├── model.json
    │   ├── metadata.json
    │   └── group1-shard1of1.bin
    │
    ├── audio/
    │   ├── model.json
    │   ├── metadata.json (si existe)
    │   └── groupX-shardXofX.bin
    │
    └── posturas/
        ├── model.json
        ├── metadata.json
        └── group1-shard1of1.bin
```

---

# 💡 ¿Cómo funciona?

### 🔸 1. Entrada (Cámara / Micrófono)
- `navigator.mediaDevices.getUserMedia({...})` para captura.
- Imagen → se dibuja en un `<canvas>` y se preprocesa.
- Audio → se captura un buffer (o se usa TM Audio) antes de predicción.

### 🔸 2. Preprocesamiento
- **Imágenes/Posturas:** resize a 224×224 + normalización.
- **Audio:**  
  - Si usas TM Audio → TF.js procesa el audio internamente.  
  - Si usas tu modelo CNN → espectrograma generado antes de entrenar.

### 🔸 3. Inferencia con TensorFlow.js
Se cargan los modelos así:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@teachablemachine/audio@0.8/dist/teachablemachine-audio.min.js"></script>
```

### 🔸 4. UI en tiempo real
- Top-1 (clase predicha)
- Top-3 probabilidades con barras
- Latencia real o simulada
- Descripción del modelo (entrenamiento, limitaciones)

---

# 🧪 Entrenamiento de modelos

### 📷 Imágenes / Posturas (Teachable Machine)
- Grabación de dataset directamente desde TM.
- Entrenamiento rápido sin código.
- Exportación automática a TF.js.

### 🎧 Audio (CNN desde Colab)
- Dataset creado desde múltiples idiomas (Kaggle).
- Limpieza, segmentación y normalización automática.
- Conversión a **log-mel espectrogramas**.
- Entrenamiento con Keras:
  ```python
  model = tf.keras.Sequential([...])
  ```
- Exportación final:
  ```python
  tfjs.converters.save_keras_model(model, "./audio_tfjs")
  ```

---

# 💾 Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/tuusuario/ai-lab
cd ai-lab
```

2. Verifica que la carpeta `models/` contenga subcarpetas:

```
models/imagenes/
models/audio/
models/posturas/
```

3. Abre `index.html` en tu navegador.  
(No requiere servidor.)

4. Para GitHub Pages:

```
Settings → Pages → Deploy from branch → main → /(root)
```

---

# 🧩 Requisitos
- Navegador moderno (Chrome recomendado)
- HTTPS para usar micrófono/cámara
- TensorFlow.js cargado desde CDN

---

# 🛠️ Tecnologías
- TensorFlow.js  
- Teachable Machine  
- JavaScript puro (sin frameworks)  
- HTML5 + Canvas  
- WebRTC (`getUserMedia`)  
- Google Colab (audio)

---

# 🏅 Créditos
Proyecto desarrollado por:

**Eder Hernández – IA Vision Coders**  
Entrenamiento, preprocesamiento, arquitectura, UI, integración con TF.js y modelos multimodales.

Enfocado en la enseñanza práctica de machine learning a estudiantes y desarrolladores.

---

# 📄 Licencia
MIT — libre para uso educativo y proyectos escolares.

---

# 🎯 Mejoras Futuras
- Detección en batch  
- Reconocimiento de gestos  
- Grabación del dataset desde la web  
- Exportar a ONNX o WebGPU  
- Dashboard más avanzado  