# NLP-II: cuadernos docentes de procesamiento del lenguaje natural con Transformers y LLM

## Descripción
Este repositorio reúne cuadernos Jupyter para la docencia práctica de la asignatura `NLP-II`. El conjunto cubre tareas de procesamiento del lenguaje natural basadas en modelos Transformer y modelos generativos, con materiales orientados a entrenamiento, evaluación, comparación de estrategias y experimentación reproducible.

Los temas detectados en los notebooks incluyen clasificación de sentimiento, inferencia textual, detección de paráfrasis, traducción automática neuronal, sistemas de preguntas y respuestas, resumen automático, generación causal, chatbots, Retrieval-Augmented Generation (RAG), destilación de conocimiento y cuantización posentrenamiento.

## Finalidad docente
La función principal del repositorio es servir como material de apoyo para asignaturas de PLN.

- Proporciona ejercicios guiados mediante notebooks `STUDENT`, con celdas incompletas y marcas `TODO`.
- Ofrece soluciones o versiones de referencia mediante notebooks `SOLVED` o `RESOLVED`.
- Facilitar la reutilización modular por temas.
- Apoyar la preparación docente, la evaluación continua y el trabajo práctico guiado.

## Contenidos
El repositorio contiene los siguientes tipos de materiales:

- Notebooks de prácticas sobre tareas clásicas de PLN con Transformers.
- Notebooks de temas avanzados relacionados con RAG, destilación y cuantización.
- Versiones diferenciadas para alumnado y profesorado en varias prácticas.
- Un script auxiliar, `create_dialog_chatbot_notebooks.py`, para generar notebooks de diálogo/chatbot.
- Metadatos de empaquetado en `pyproject.toml`.
- Un archivo de licencia `LICENSE.txt`.
- Directorios auxiliares `build/` y `nlp_ii.egg-info/`, presentes en la raíz del repositorio.

Estructura observada en la raíz del repositorio:

```text
.
├── *.ipynb
├── create_dialog_chatbot_notebooks.py
├── LICENSE.txt
├── pyproject.toml
├── README.md
├── build/
└── nlp_ii.egg-info/
```

## Requisitos
Requisitos verificables a partir de `pyproject.toml`:

- Python `>=3.10`
- `torch>=2.2.0`
- `transformers==4.56.2`
- `datasets==2.19.1`
- `accelerate==1.10.1`
- `evaluate==0.4.1`
- `sentencepiece>=0.1.99`
- `scikit-learn>=1.5.0`
- `numpy==2.0.2`
- `pandas>=2.2.0`
- `matplotlib>=3.8.0`
- `huggingface-hub>=0.25.2`
- `kagglehub>=0.2.5`
- `nltk>=3.8.1`
- `fsspec==2024.3.1`
- `pyarrow-hotfix`
- `bitsandbytes>=0.43.0` solo cuando el sistema no es Windows, según la condición declarada en el propio `pyproject.toml`

Dependencias adicionales en notebooks concretos :

- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `psutil`
- `tqdm`
- `sacremoses` como dependencia recomendada por MarianMT en una ejecución observada

## Instalación
La referencia principal para instalar dependencias es `pyproject.toml`.

Instalación base recomendada en entorno Unix o compatible con bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install jupyterlab
```

Si se prevé ejecutar todos los notebooks, incluidas las prácticas de RAG, destilación y benchmarking, conviene instalar también las dependencias adicionales detectadas:

```bash
pip install sentence-transformers faiss-cpu rank-bm25 psutil tqdm sacremoses
```

En entornos con GPU y sistemas no Windows, puede ser útil completar la instalación opcional de cuantización:

```bash
pip install bitsandbytes
```

## Uso
Tras activar el entorno, puede iniciarse Jupyter Lab o Jupyter Notebook:

```bash
jupyter lab
```

Uso docente recomendado:

- Abrir primero los notebooks `STUDENT` cuando la finalidad sea práctica guiada o evaluación.
- Reservar los notebooks `SOLVED` o `RESOLVED` para preparación del profesorado, corrección, autoestudio posterior o revisión de soluciones.
- Utilizar los notebooks sin sufijo `STUDENT` o `SOLVED/RESOLVED` como materiales autónomos, normalmente de carácter más avanzado o monográfico.
- Revisar siempre las primeras celdas de cada notebook, ya que varios materiales incluyen instrucciones específicas para Google Colab, reinicio del entorno o instalación puntual de bibliotecas.

## Organización de notebooks
Convenciones observadas:

- `STUDENT`: versión para alumnado, con celdas incompletas, guías o `TODO`.
- `SOLVED`: versión resuelta o de referencia.

Tabla de notebooks detectados en la raíz del repositorio:

| Archivo | Tipo | Tema u objetivo verificable | Observaciones |
|---|---|---|---|
| `BERT_IMDB_SOLVED.ipynb` | SOLVED | Ajuste fino de BERT para clasificación de sentimiento binaria sobre IMDb | Usa `kagglehub`; orientado a Colab |
| `BERT_IMDB_STUDENT.ipynb` | STUDENT | Clasificación de sentimiento en IMDb con BERT | Incluye celdas `TODO` |
| `DIALOG_Chatbot_SOLVED_EN.ipynb` | SOLVED | Construcción de un mini-chatbot con modelo decoder-only | Material en inglés |
| `DIALOG_Chatbot_STUDENT_EN.ipynb` | STUDENT | Chatbot con modelo decoder-only | Material en inglés con `TODO` |
| `Generative_Homebrewing_RESOLVED.ipynb` | RESOLVED | Laboratorio generativo con LLM sobre homebrewing | Incluye parafraseo, recetas, chat y clasificación por prompting |
| `Generative_Homebrewing_STUDENT.ipynb` | STUDENT | Laboratorio generativo con LLM sobre homebrewing | Plantilla con `TODO`; literales de ejemplo en español |
| `GPT2_Resolved.ipynb` | RESOLVED | Clínica de decodificación para generación creativa en español con GPT-2 | Analiza calidad, diversidad, estabilidad y latencia |
| `GPT2_Student.ipynb` | STUDENT | Clínica de decodificación con GPT-2 | Requiere completar celdas y gestionar token de Hugging Face |
| `Knowledge_Distillation_Text_Classification.ipynb` | Único | Destilación de conocimiento para clasificación de texto | Mide exactitud, latencia y memoria |
| `NMT_Marian_ES_EN_SOLVED.ipynb` | SOLVED | Traducción automática español-inglés con MarianMT | Compara `greedy`, `beam search` y `sampling` |
| `NMT_Marian_ES_EN_STUDENT.ipynb` | STUDENT | Traducción automática español-inglés con MarianMT | `TODO` redactados en español |
| `Post_Training_Quantization_and_Benchmarking (1).ipynb` | Único | Cuantización posentrenamiento para inferencia con LLM | Compara FP16, 8-bit y 4-bit; nombre de archivo no normalizado |
| `QA_Transformers_SOLVED.ipynb` | SOLVED | Sistemas de preguntas y respuestas con Transformers | Incluye mini QA open-domain con recuperación simple |
| `QA_Transformers_STUDENT.ipynb` | STUDENT | Sistemas de preguntas y respuestas con Transformers | `TODO` redactados en español |
| `RAG_basics_and_evaluation.ipynb` | Único | Fundamentos de RAG y evaluación básica | Emplea embeddings y FAISS |
| `RAG_production_hybrid.ipynb` | Único | RAG híbrido con reranking, filtrado por metadatos y defensas frente a inyección | Material avanzado de orientación aplicada |
| `SUMMARY_Extractive_Abstractive_SOLVED.ipynb` | SOLVED | Resumen automático extractivo y abstractivo | Incluye una métrica ROUGE-1 simplificada |
| `SUMMARY_Extractive_Abstractive_STUDENT.ipynb` | STUDENT | Resumen automático extractivo y abstractivo | `TODO` redactados en español |
| `XLMR_Paraphrase_SOLVED.ipynb` | SOLVED | Detección de paráfrasis en PAWS-X español con XLM-RoBERTa | Ajuste fino y evaluación |
| `XLMR_Paraphrase_STUDENT.ipynb` | STUDENT | Detección de paráfrasis con XLM-RoBERTa | Material en inglés con `TODO` |
| `XNLI_BETO_SOLVED.ipynb` | SOLVED | Inferencia textual en español con BETO sobre XNLI | Incluye matriz de confusión y análisis de errores |
| `XNLI_BETO_STUDENT.ipynb` | STUDENT | Inferencia textual en español con BETO sobre XNLI | Versión guiada con instalación explícita para Colab |

## Recomendaciones de uso docente
- Para estudiantes: comenzar por la versión `STUDENT` correspondiente y completar las celdas en el orden propuesto, sin saltar las secciones de configuración y carga de datos.
- Para estudiantes: conservar evidencias de ejecución, métricas y observaciones, ya que varios notebooks incluyen fases de análisis de errores, comparación de estrategias o discusión de resultados.
- Para estudiantes: no asumir que todos los notebooks comparten exactamente las mismas dependencias; conviene revisar la primera sección de cada cuaderno antes de ejecutarlo.
- Para profesorado: utilizar las versiones `SOLVED/RESOLVED` como solución de referencia, material de apoyo en seminario o base para diseñar rúbricas y entregables.
- Para profesorado: valorar una secuenciación por bloques, por ejemplo clasificación y NLI, después traducción/QA/resumen y finalmente generación, RAG y optimización de inferencia.

## Notas técnicas
- `pyproject.toml` sí declara una base de dependencias y fija `Python >= 3.10`.
- Varios notebooks incluyen salidas ejecutadas y metadatos de Google Colab, por lo que el repositorio no contiene exclusivamente cuadernos "limpios".
- Algunos notebooks son explícitamente "GPU-first" o están optimizados para Colab, aunque varios declaran también rutas de ejecución en CPU.
- Las prácticas de cuantización y algunos notebooks generativos usan o intentan usar `bitsandbytes`; en CPU o en Windows puede ser necesario recurrir a las rutas alternativas previstas por los propios cuadernos.
- Las prácticas RAG y de benchmarking importan bibliotecas adicionales que no están reflejadas de forma uniforme en la configuración base del proyecto.

## Licencia
El repositorio incluye el archivo [`LICENSE.txt`](LICENSE.txt), cuyo encabezado corresponde a la **GNU General Public License, versión 3 (GPL-3.0)**.
