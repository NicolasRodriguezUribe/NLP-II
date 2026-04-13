# NLP-II: cuadernos docentes de procesamiento del lenguaje natural con Transformers y LLM

## Introducción
Autor: Nicolás Rodríguez Uribe
Fecha de última actualización: 13/04/2026
Material docente en abierto de la Universidad Rey Juan Carlos
Asignatura: Procesamiento del Lenguaje Natural II
Grado en Inteligencia Artificial
Licencia: **GNU General Public License, versión 3 (GPL-3.0)**.
Depósito URJC: https://burjcdigital.urjc.es/items/e7f1b47d-be39-4e11-ab68-487bc9278099

## Descripción
Este repositorio reúne cuadernos Jupyter para la docencia práctica de la asignatura `NLP-II`. El conjunto cubre tareas de procesamiento del lenguaje natural basadas en modelos Transformer y modelos generativos, con materiales orientados a entrenamiento, evaluación, comparación de estrategias y experimentación reproducible.

Los temas en los notebooks incluyen clasificación de sentimiento, inferencia textual, detección de paráfrasis, traducción automática neuronal, sistemas de preguntas y respuestas, resumen automático, generación causal, chatbots, Retrieval-Augmented Generation (RAG), destilación de conocimiento y cuantización posentrenamiento.

## Finalidad docente
La función principal del repositorio es servir como material de apoyo para asignaturas de PLN.

- Proporciona ejercicios guiados mediante notebooks `STUDENT`, con celdas incompletas y marcas `TODO`.
- Ofrece soluciones o versiones de referencia mediante notebooks `SOLVED` o `RESOLVED`.
- Facilita la reutilización modular por temas.
- Apoya la preparación docente, la evaluación continua y el trabajo práctico guiado.

## Contenidos
El repositorio contiene los siguientes tipos de materiales:

- Notebooks de prácticas sobre tareas clásicas de PLN con Transformers.
- Notebooks de temas avanzados relacionados con RAG, destilación y cuantización.
- Versiones diferenciadas para alumnado y profesorado en varias prácticas.
- Un script auxiliar en `scripts/create_dialog_chatbot_notebooks.py` para generar notebooks de diálogo/chatbot.
- Metadatos de empaquetado en `pyproject.toml`.
- Un archivo de licencia `LICENSE`.

Estructura del repositorio:

```text
.
├── LICENSE
├── notebooks/
│   ├── advanced/
│   ├── classification/
│   ├── generation_dialogue/
│   ├── nli_paraphrase/
│   ├── qa_summarization/
│   └── translation/
├── pyproject.toml
├── scripts/
│   └── create_dialog_chatbot_notebooks.py
└── README.md
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
- `bitsandbytes>=0.43.0` solo cuando el sistema no es Windows, según `pyproject.toml`

Dependencias adicionales en notebooks concretos:

- `sentence-transformers`
- `faiss-cpu`
- `rank-bm25`
- `psutil`
- `tqdm`
- `sacremoses` como dependencia recomendada por MarianMT

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
jupyter lab notebooks
```

## Organización de notebooks
Convenciones observadas:

- `STUDENT`: versión para alumnado, con celdas incompletas, guías o `TODO`.
- `SOLVED`: versión resuelta o de referencia.
- `RESOLVED` o `Resolved`: variante equivalente a versión resuelta.

Carpetas temáticas utilizadas:

- `notebooks/classification`: clasificación de sentimiento.
- `notebooks/nli_paraphrase`: inferencia textual y detección de paráfrasis.
- `notebooks/translation`: traducción automática neuronal.
- `notebooks/qa_summarization`: preguntas y respuestas y resumen automático.
- `notebooks/generation_dialogue`: generación causal, chatbots y prácticas de diálogo.
- `notebooks/advanced`: RAG, destilación de conocimiento y cuantización.

Tabla de notebooks organizados por carpeta:

| Ruta | Tipo | Tema u objetivo verificable | Observaciones |
|---|---|---|---|
| `notebooks/classification/BERT_IMDB_SOLVED.ipynb` | SOLVED | Ajuste fino de BERT para clasificación de sentimiento binaria sobre IMDb | Usa `kagglehub`; orientado a Colab |
| `notebooks/classification/BERT_IMDB_STUDENT.ipynb` | STUDENT | Clasificación de sentimiento en IMDb con BERT | Incluye celdas `TODO` |
| `notebooks/generation_dialogue/DIALOG_Chatbot_SOLVED_EN.ipynb` | SOLVED | Construcción de un mini-chatbot con modelo decoder-only | Material en inglés |
| `notebooks/generation_dialogue/DIALOG_Chatbot_STUDENT_EN.ipynb` | STUDENT | Chatbot con modelo decoder-only | Material en inglés con `TODO` |
| `notebooks/generation_dialogue/Generative_Homebrewing_RESOLVED.ipynb` | RESOLVED | Laboratorio generativo con LLM sobre homebrewing | Incluye parafraseo, recetas, chat y clasificación por prompting |
| `notebooks/generation_dialogue/Generative_Homebrewing_STUDENT.ipynb` | STUDENT | Laboratorio generativo con LLM sobre homebrewing | Plantilla con `TODO`; literales de ejemplo en español |
| `notebooks/generation_dialogue/GPT2_Resolved.ipynb` | RESOLVED | Clínica de decodificación para generación creativa en español con GPT-2 | Analiza calidad, diversidad, estabilidad y latencia |
| `notebooks/generation_dialogue/GPT2_Student.ipynb` | STUDENT | Clínica de decodificación con GPT-2 | Requiere completar celdas y gestionar token de Hugging Face |
| `notebooks/advanced/Knowledge_Distillation_Text_Classification.ipynb` | Único | Destilación de conocimiento para clasificación de texto | Mide exactitud, latencia y memoria |
| `notebooks/translation/NMT_Marian_ES_EN_SOLVED.ipynb` | SOLVED | Traducción automática español-inglés con MarianMT | Compara `greedy`, `beam search` y `sampling` |
| `notebooks/translation/NMT_Marian_ES_EN_STUDENT.ipynb` | STUDENT | Traducción automática español-inglés con MarianMT | `TODO` redactados en español |
| `notebooks/advanced/Post_Training_Quantization_and_Benchmarking.ipynb` | Único | Cuantización posentrenamiento para inferencia con LLM | Compara FP16, 8-bit y 4-bit |
| `notebooks/qa_summarization/QA_Transformers_SOLVED.ipynb` | SOLVED | Sistemas de preguntas y respuestas con Transformers | Incluye mini QA open-domain con recuperación simple |
| `notebooks/qa_summarization/QA_Transformers_STUDENT.ipynb` | STUDENT | Sistemas de preguntas y respuestas con Transformers | `TODO` redactados en español |
| `notebooks/advanced/RAG_basics_and_evaluation.ipynb` | Único | Fundamentos de RAG y evaluación básica | Emplea embeddings y FAISS |
| `notebooks/advanced/RAG_production_hybrid.ipynb` | Único | RAG híbrido con reranking, filtrado por metadatos y defensas frente a inyección | Material avanzado de orientación aplicada |
| `notebooks/qa_summarization/SUMMARY_Extractive_Abstractive_SOLVED.ipynb` | SOLVED | Resumen automático extractivo y abstractivo | Incluye una métrica ROUGE-1 simplificada |
| `notebooks/qa_summarization/SUMMARY_Extractive_Abstractive_STUDENT.ipynb` | STUDENT | Resumen automático extractivo y abstractivo | `TODO` redactados en español |
| `notebooks/nli_paraphrase/XLMR_Paraphrase_SOLVED.ipynb` | SOLVED | Detección de paráfrasis en PAWS-X español con XLM-RoBERTa | Ajuste fino y evaluación |
| `notebooks/nli_paraphrase/XLMR_Paraphrase_STUDENT.ipynb` | STUDENT | Detección de paráfrasis con XLM-RoBERTa | Material en inglés con `TODO` |
| `notebooks/nli_paraphrase/XNLI_BETO_SOLVED.ipynb` | SOLVED | Inferencia textual en español con BETO sobre XNLI | Incluye matriz de confusión y análisis de errores |
| `notebooks/nli_paraphrase/XNLI_BETO_STUDENT.ipynb` | STUDENT | Inferencia textual en español con BETO sobre XNLI | Versión guiada con instalación explícita para Colab |

## Recomendaciones para estudiantes
- Comenzar por la versión `STUDENT` correspondiente y completar las celdas en el orden propuesto, sin saltar las secciones de configuración y carga de datos.
- Conservar evidencias de ejecución, métricas y observaciones, ya que varios notebooks incluyen fases de análisis de errores, comparación de estrategias o discusión de resultados.
- No asumir que todos los notebooks comparten exactamente las mismas dependencias; conviene revisar la primera sección de cada cuaderno antes de ejecutarlo.

## Notas técnicas
- `pyproject.toml` declara una base de dependencias y fija `Python >= 3.10`.
- Varios notebooks incluyen salidas ejecutadas y metadatos de Google Colab, por lo que el repositorio no contiene exclusivamente cuadernos "limpios".
- Algunos notebooks son explícitamente "GPU-first" o están optimizados para Colab, aunque varios declaran también rutas de ejecución en CPU.
- Las prácticas de cuantización y algunos notebooks generativos usan o intentan usar `bitsandbytes`; en CPU o en Windows puede ser necesario recurrir a las rutas alternativas previstas por los propios cuadernos.
- Las prácticas RAG y de benchmarking importan bibliotecas adicionales que no están reflejadas de forma uniforme en la configuración base del proyecto.

## Licencia
El repositorio incluye el archivo [`LICENSE`](LICENSE), correspondiente a la **GNU General Public License, versión 3 (GPL-3.0)**.
