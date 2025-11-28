// app.js
// ----------------------------------------------------
// Lógica principal del laboratorio de IA multimodal
// Demo educativa lista para GitHub Pages (sin bundlers)
// ----------------------------------------------------
let audioChunks = []; // almacena los chunks grabados

// Rutas de modelos (ajusta las rutas según tu estructura real)
const MODEL_PATHS = {
  images: "models/imagenes/model.json",
  audio: "models/audio/model.json",
  postures: "models/posturas/model.json",
};

// Etiquetas de clases (labels) para cada modelo
// *** IMPORTANTE: respeta exactamente los nombres pedidos ***
const LABELS = {
  // Modelo 1 – Imágenes (accesorios en el rostro)
  images: ["con_gorra", "con_tapabocas", "con_gafas", "sin_accesorios"],

  // Modelo 2 – Audio (idiomas)
  audio: ["french","english","spanish","german"],

  // Modelo 3 – Imágenes/Posturas (acciones tipo pelea/deporte)z
  postures: ["puño","ataque","boxeo","patada"],
};

// Almacenamos los modelos en memoria para reutilizarlos
const models = {
  images: null,
  audio: null,
  postures: null,
};

// Estado general de la UI / medios
let currentMode = "images"; // modo inicial
let currentStream = null; // stream de video usado por imágenes/posturas
let visionIntervalId = null; // intervalo de inferencia para visión

// Estado para audio (micrófono)
let audioStream = null;
let audioRecorder = null;
let isRecordingAudio = false;

// ----------------------------
// Inicialización de la página
// ----------------------------

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();

  initImagesMode();
  initAudioMode();
  initPosturesMode();

  // Inicializa listas Top-3 con placeholders
  initTop3Placeholders();
});

// ----------------------------
// Pestañas / modos
// ----------------------------

/**
 * Configura los eventos de los botones de pestañas (Imágenes, Sonidos, Posturas)
 */
function setupTabs() {
  const tabButtons = document.querySelectorAll(".tab-button");
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetMode = btn.getAttribute("data-mode");
      if (!targetMode || targetMode === currentMode) return;

      // Cambiar pestaña activa
      tabButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Cambiar panel visible
      switchMode(targetMode);
    });
  });
}

/**
 * Cambia el modo activo, deteniendo medios del modo anterior.
 * @param {string} mode "images" | "audio" | "postures"
 */
function switchMode(mode) {
  // Detiene lo que esté activo en el modo actual
  stopMediaForMode(currentMode);

  // Oculta todos los paneles y muestra el nuevo
  document.querySelectorAll(".mode-panel").forEach((panel) => {
    panel.classList.remove("active");
  });
  const targetPanel = document.getElementById(`mode-${mode}`);
  if (targetPanel) {
    targetPanel.classList.add("active");
  }

  currentMode = mode;
}

// ----------------------------
// Inicialización por modo
// ----------------------------

/**
 * Modo Imágenes: configura eventos de botones y subida de archivo.
 */
function initImagesMode() {
  const startBtn = document.getElementById("images-start-btn");
  const stopBtn = document.getElementById("images-stop-btn");
  const uploadInput = document.getElementById("images-upload");

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      await startVisionPipeline("images");
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      stopMediaForMode("images");
    });
  }

  if (uploadInput) {
    uploadInput.addEventListener("change", (event) => {
      handleImageUpload("images", event.target.files);
      // Limpia el input para permitir subir el mismo archivo otra vez
      event.target.value = "";
    });
  }
}

/**
 * Modo Audio: configura grabación y subida de audio.
 */
function initAudioMode() {
  const recordBtn = document.getElementById("audio-record-btn");
  const stopBtn = document.getElementById("audio-stop-btn");
  const uploadInput = document.getElementById("audio-upload");

  if (recordBtn) {
    recordBtn.addEventListener("click", () => {
      startAudioRecording();
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      stopAudioRecording();
    });
  }

  if (uploadInput) {
    uploadInput.addEventListener("change", async (event) => {
      const files = event.target.files;
      await handleAudioUpload(files);
      event.target.value = "";
    });
  }
}

/**
 * Modo Posturas: similar al de Imágenes pero con otro modelo y labels.
 */
function initPosturesMode() {
  const startBtn = document.getElementById("postures-start-btn");
  const stopBtn = document.getElementById("postures-stop-btn");
  const uploadInput = document.getElementById("postures-upload");

  if (startBtn) {
    startBtn.addEventListener("click", async () => {
      await startVisionPipeline("postures");
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      stopMediaForMode("postures");
    });
  }

  if (uploadInput) {
    uploadInput.addEventListener("change", (event) => {
      handleImageUpload("postures", event.target.files);
      event.target.value = "";
    });
  }
}

// ----------------------------
// Carga de modelos
// ----------------------------

/**
 * Carga el modelo de un modo si aún no está en memoria.
 * Si falla, se continuará con simulación de predicciones.
 * @param {string} mode
 */
async function loadModelIfNeeded(mode) {
  if (models[mode]) return models[mode];

  const path = MODEL_PATHS[mode];
  if (!path || !window.tf) {
    console.warn("TensorFlow.js o ruta de modelo no disponible.");
    return null;
  }

  try {
    // Teachable Machine suele exportar modelos compatibles con loadLayersModel
    const model = await tf.loadLayersModel(path);
    models[mode] = model;
    console.log(`Modelo "${mode}" cargado desde: ${path}`);
    return model;
  } catch (err) {
    console.warn(
      `No se pudo cargar el modelo "${mode}" desde ${path}. Se usará simulación de predicciones.`,
      err
    );
    models[mode] = null;
    return null;
  }
}

// ----------------------------
// Visión: webcam + inferencia
// ----------------------------

/**
 * Inicia la pipeline de visión para un modo (images o postures):
 * - Carga modelo
 * - Solicita permiso de cámara
 * - Arranca bucle de inferencia periódica
 */
async function startVisionPipeline(mode) {
    
  // Carga el modelo (si existe). Si falla, igualmente se va a usar simulación.
  await loadModelIfNeeded(mode);

  const latencyEl = document.getElementById(`${mode}-latency`);
  if (latencyEl) {
    latencyEl.textContent = "Latencia estimada: esperando cámara...";
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Este navegador no soporta acceso a la cámara (getUserMedia).");
    return;
  }

  try {
    // Primero queda cualquier stream previo
    stopMediaForMode(mode);

    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    currentStream = stream;

    const videoEl = document.getElementById(`${mode}-video`);
    if (videoEl) {
      videoEl.srcObject = stream;
      await videoEl.play();
    }

    if (latencyEl) {
      latencyEl.textContent = "Latencia estimada: capturando frames...";
    }

    // Bucle de inferencia cada ~1.2 segundos
    visionIntervalId = setInterval(() => {




  runVisionInference(mode);

}, 1200);
  } catch (err) {
    console.error("Error al acceder a la cámara:", err);
    alert("No se pudo acceder a la cámara. Revisa los permisos del navegador.");
  }
}

/**
 * Realiza una inferencia sobre el frame actual de video.
 * @param {string} mode
 */
function runVisionInference(mode) {
    
  const videoEl = document.getElementById(`${mode}-video`);
  const canvasEl = document.getElementById(`${mode}-canvas`);
  const labels = LABELS[mode];

  if (!videoEl || !canvasEl || !labels || labels.length === 0) return;
  if (videoEl.readyState < 2) {
    // El video aún no tiene datos suficientes
    return;
  }

  const ctx = canvasEl.getContext("2d");
  // Ajusta el canvas al tamaño real del video
  canvasEl.width = videoEl.videoWidth || 320;
  canvasEl.height = videoEl.videoHeight || 240;

  // Dibujamos el frame actual en el canvas
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  const t0 = performance.now();
console.log("Modelo:", models[mode]);
  // Si el modelo no está cargado, simplemente simulamos probabilidades
  if (!models[mode]) {
    const probs = randomProbabilities(labels.length);
    const latencySim = randomInRange(80, 220);
    setLatencyText(mode, latencySim);
    updatePredictions(mode, probs, labels);
    return;
  }

  // Si el modelo está cargado, aplicamos una inferencia real.
  // NOTA: asumimos que el modelo espera un input 224x224x3 normalizado [0, 1].
  let probs = [];

  try {
    tf.tidy(() => {
      const inputSize = 192;
      const imgTensor = tf.browser
        .fromPixels(canvasEl)
        .resizeBilinear([inputSize, inputSize])
        .toFloat()
        .div(255)
        .expandDims(0); // shape: [1, 224, 224, 3]

      const logits = models[mode].predict(imgTensor);
      console.log("Predict output:", logits);

      const probsTensor = tf.softmax(logits);
      probs = Array.from(probsTensor.dataSync());
      console.log("Probabilidades:", probs);
    });

    const t1 = performance.now();
    const latency = t1 - t0;

    setLatencyText(mode, latency);
    updatePredictions(mode, probs, labels);
  } catch (err) {
    console.warn(
      "Error durante inferencia de visión, usando simulación de probabilidades:",
      err
    );
    const probsFallback = randomProbabilities(labels.length);
    const latencySim = randomInRange(80, 220);
    setLatencyText(mode, latencySim);
    updatePredictions(mode, probsFallback, labels);
  }
}

/**
 * Maneja subida de imagen para un modo de visión (images o postures).
 * Carga la imagen en el canvas y realiza una inferencia única.
 */
async function handleImageUpload(mode, files) {
  if (!files || files.length === 0) return;

  const file = files[0];
  if (!file.type.startsWith("image/")) {
    alert("Por favor, sube un archivo de imagen válido.");
    return;
  }

  const labels = LABELS[mode];
  const canvasEl = document.getElementById(`${mode}-canvas`);
  if (!canvasEl) return;

  const ctx = canvasEl.getContext("2d");
  const img = new Image();
  img.onload = async () => {
    // Ajustamos el canvas a la imagen cargada
    canvasEl.width = img.width;
    canvasEl.height = img.height;
    ctx.drawImage(img, 0, 0);

    const t0 = performance.now();

    // Aseguramos que el modelo esté cargado
    await loadModelIfNeeded(mode);

    if (!models[mode]) {
      // Si falla la carga, simulamos predicciones
      const probsSim = randomProbabilities(labels.length);
      const latencySim = randomInRange(80, 260);
      setLatencyText(mode, latencySim);
      updatePredictions(mode, probsSim, labels);
      return;
    }

    let probs = [];
    try {
      tf.tidy(() => {
        const inputSize = 224;
        const imgTensor = tf.browser
          .fromPixels(canvasEl)
          .resizeBilinear([inputSize, inputSize])
          .toFloat()
          .div(255)
          .expandDims(0);

        const logits = models[mode].predict(imgTensor);
        const probsTensor = tf.softmax(logits);
        probs = Array.from(probsTensor.dataSync());
      });

      const t1 = performance.now();
      const latency = t1 - t0;

      setLatencyText(mode, latency);
      updatePredictions(mode, probs, labels);
    } catch (err) {
      console.warn(
        "Error durante inferencia al subir imagen, usando simulación:",
        err
      );
      const probsFallback = randomProbabilities(labels.length);
      const latencySim = randomInRange(80, 260);
      setLatencyText(mode, latencySim);
      updatePredictions(mode, probsFallback, labels);
    }
  };

  img.onerror = () => {
    alert("No se pudo cargar la imagen. Intenta con otro archivo.");
  };

  // Convertimos el archivo a data URL para cargarlo en el objeto Image
  const reader = new FileReader();
  reader.onload = (e) => {
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// ----------------------------
// Audio: grabación + simulación
// ----------------------------

/**
 * Inicia la grabación de audio.
 * Esta implementación usa MediaRecorder, pero la inferencia
 * se simula con probabilidades aleatorias. El código está
 * preparado para que en el futuro se reemplace por extracción
 * de características reales + modelo de audio en TF.js.
 */
async function startAudioRecording() {
  if (isRecordingAudio) return;

  const latencyEl = document.getElementById("audio-latency");
  if (latencyEl) {
    latencyEl.textContent = "Latencia estimada: esperando micrófono...";
  }

  try {
    if (!audioStream) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Este navegador no soporta acceso al micrófono.");
        return;
      }

      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });
    }

    // Inicializamos MediaRecorder solo una vez
if (!audioRecorder) {
  audioRecorder = new MediaRecorder(audioStream);

  // Cada fragmento de audio grabado se acumula en audioChunks
  audioRecorder.addEventListener("dataavailable", (e) => {
    audioChunks.push(e.data);
  });

  // Cuando termina la grabación, creamos un blob reproducible
  audioRecorder.addEventListener("stop", async () => {
    isRecordingAudio = false;

    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    audioChunks = []; // limpiamos para la próxima grabación

    // Creamos la URL reproducible
    const audioURL = URL.createObjectURL(audioBlob);

    // Actualizamos el reproductor en el DOM
    const playbackContainer = document.getElementById("audio-playback-container");
    const audioPlayer = document.getElementById("audio-playback");

    if (audioPlayer) {
      audioPlayer.src = audioURL;
      playbackContainer.classList.remove("hidden");
    }

    // Luego hacemos la "inferencia" simulada
    await runAudioInferenceSimulated();
  });
}


    // Carga lazily el modelo (aunque luego simulemos la entrada)
    await loadModelIfNeeded("audio");

    // Arrancamos la grabación
    audioRecorder.start();
    isRecordingAudio = true;

    if (latencyEl) {
      latencyEl.textContent =
        "Latencia estimada: grabando clip (1–3 s) para análisis...";
    }
  } catch (err) {
    console.error("Error al acceder al micrófono:", err);
    alert(
      "No se pudo acceder al micrófono. Revisa permisos o la configuración de tu dispositivo."
    );
  }
}

/**
 * Detiene la grabación de audio si está activa.
 */
function stopAudioRecording() {
  if (audioRecorder && isRecordingAudio) {
    audioRecorder.stop();
  }
}

/**
 * Simula la inferencia de audio: genera probabilidades aleatorias
 * y actualiza la UI como si el modelo hubiera procesado un clip.
 * En un escenario real, aquí se tomaría el buffer de audio,
 * se calcularían features (por ejemplo espectrogramas) y se
 * ejecutaría el modelo de TF.js.
 */
async function runAudioInferenceSimulated() {
  const labels = LABELS.audio;
  if (!labels || labels.length === 0) return;

  // Simulamos un pequeño tiempo de "procesamiento"
  const simulatedProcessing = randomInRange(200, 600);
  await new Promise((resolve) => setTimeout(resolve, simulatedProcessing));

  const probs = randomProbabilities(labels.length);
  const latency = randomInRange(150, 400);
  setLatencyText("audio", latency);
  updatePredictions("audio", probs, labels);
}

/**
 * Maneja subida de audio. Igual que la grabación, aquí solo simulamos
 * la inferencia, pero el código está preparado para que se conecte
 * con un modelo real de audio en el futuro.
 */
async function handleAudioUpload(files) {
  if (!files || files.length === 0) return;

  const file = files[0];
  if (!file.type.startsWith("audio/")) {
    alert("Por favor, sube un archivo de audio válido (.wav/.mp3, etc.).");
    return;
  }

  // Carga del modelo (por si queremos usarlo en el futuro)
  await loadModelIfNeeded("audio");

  const latencyEl = document.getElementById("audio-latency");
  if (latencyEl) {
    latencyEl.textContent = "Latencia estimada: analizando archivo de audio...";
  }

  // Simulamos "leer" el archivo y procesarlo
  await new Promise((resolve) => setTimeout(resolve, randomInRange(250, 700)));

  const labels = LABELS.audio;
  const probs = randomProbabilities(labels.length);
  const latency = randomInRange(170, 420);
  setLatencyText("audio", latency);
  updatePredictions("audio", probs, labels);
}

// ----------------------------
// Detener medios según modo
// ----------------------------

/**
 * Detiene streams y timers activos asociados a un modo.
 * @param {string} mode
 */
function stopMediaForMode(mode) {
  // Detener inferencia de visión y liberar cámara
  if (visionIntervalId) {
    clearInterval(visionIntervalId);
    visionIntervalId = null;
  }

  if (currentStream) {
    currentStream.getTracks().forEach((track) => track.stop());
    currentStream = null;
  }

  // Detener audio si estamos en modo audio
  if (mode === "audio") {
    if (audioRecorder && isRecordingAudio) {
      audioRecorder.stop();
    }
    if (audioStream) {
      audioStream.getTracks().forEach((track) => track.stop());
      audioStream = null;
    }
  }
}

// ----------------------------
// Actualización de UI (preds)
// ----------------------------

/**
 * Inicializa las listas de Top-3 con barras vacías como placeholder.
 * Útil para que los estudiantes vean la estructura antes de la primera inferencia.
 */
function initTop3Placeholders() {
  ["images", "audio", "postures"].forEach((mode) => {
    const labels = LABELS[mode];
    if (!labels) return;
    const probs = new Array(labels.length).fill(0);
    updatePredictions(mode, probs, labels);
  });
}

/**
 * Actualiza la UI de Top-1 y Top-3 para un modo dado.
 * @param {string} mode
 * @param {number[]} probs Vector de probabilidades (0–1)
 * @param {string[]} labels Etiquetas de las clases
 */
function updatePredictions(mode, probs, labels) {
  if (!probs || probs.length === 0 || !labels || labels.length === 0) return;

  // --- Top-1 ---
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  const top1Prob = probs[maxIdx];
  const top1Label = labels[maxIdx] || `Clase ${maxIdx}`;

  const top1LabelEl = document.getElementById(`${mode}-top1-label`);
  const top1PercentEl = document.getElementById(`${mode}-top1-percent`);
  if (top1LabelEl) top1LabelEl.textContent = top1Label;
  if (top1PercentEl) {
    top1PercentEl.textContent = `${(top1Prob * 100).toFixed(1)} %`;
  }

  // --- Top-3 ---
  const pairs = labels.map((label, i) => ({
    label,
    prob: probs[i] || 0,
  }));
  pairs.sort((a, b) => b.prob - a.prob);
  const top3 = pairs.slice(0, 3);

  const listEl = document.getElementById(`${mode}-top3-list`);
  if (!listEl) return;

  listEl.innerHTML = "";
  top3.forEach((item) => {
    const row = document.createElement("div");
    row.className = "top3-row";

    const percentStr = `${(item.prob * 100).toFixed(1)}%`;
    const safeWidth = Math.max(3, item.prob * 100); // que siempre se vea algo

    row.innerHTML = `
      <span class="class-name">${item.label}</span>
      <div class="bar">
        <div class="bar-fill" style="width:${safeWidth.toFixed(1)}%"></div>
      </div>
      <span class="percent">${percentStr}</span>
    `;
    listEl.appendChild(row);
  });
}

/**
 * Actualiza el texto de latencia aproximada.
 * @param {string} mode
 * @param {number} ms
 */
function setLatencyText(mode, ms) {
  const el = document.getElementById(`${mode}-latency`);
  if (!el) return;
  el.textContent = `Latencia estimada: ${ms.toFixed(0)} ms`;
}

// ----------------------------
// Utilidades varias
// ----------------------------

/**
 * Genera un vector de probabilidades aleatorias que suman 1.
 * Útil para simular salida de un modelo de clasificación.
 * @param {number} length
 * @returns {number[]}
 */
function randomProbabilities(length) {
  const arr = [];
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const v = Math.random() + 0.01; // para evitar ceros exactos
    arr.push(v);
    sum += v;
  }
  return arr.map((v) => v / sum);
}

/**
 * Devuelve un número aleatorio en el rango [min, max].
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function randomInRange(min, max) {
  return min + Math.random() * (max - min);
}
