# 📊 Guía de Evaluación de Modelos Gemma3n

Esta guía explica cómo usar el sistema de evaluación completo para los modelos Gemma3n entrenados en detección de basura marina.

## 📁 Archivos del Sistema de Evaluación

### Módulos Principales

1. **[tools/metrics.py](tools/metrics.py)** - Clases para cálculo de métricas:
   - `DetectionMetrics`: Métricas de detección basadas en IoU (Precision, Recall, F1)
   - `ClassificationMetrics`: Métricas de clasificación (presencia de clases)
   - `COCOMetrics`: Métricas estilo COCO (mAP@0.5, mAP@0.75, mAP@0.5:0.95)
   - `PerformanceMetrics`: Métricas de rendimiento (latencia, FPS, tasa de parsing)

2. **[tools/visualizations.py](tools/visualizations.py)** - Generación de visualizaciones:
   - Gráficos de barras por clase
   - Matrices de confusión
   - Curvas de Precision-Recall
   - Comparaciones entre modelos
   - Dashboard completo de evaluación
   - Reportes HTML interactivos
   - Visualizaciones de bounding boxes

3. **[tools/evaluate_models.py](tools/evaluate_models.py)** - Script principal de evaluación:
   - Clase `ModelEvaluator`: Evaluador completo de modelos
   - Soporte para múltiples conjuntos de datos
   - Comparación entre modelo base y fine-tuned
   - Generación automática de reportes

---

## 🚀 Instalación

### Opción 1: Usando Docker (RECOMENDADO)

**Todo el desarrollo, entrenamiento y evaluación se realiza dentro del contenedor Docker.**

```bash
# 1. Construir e iniciar el contenedor
docker-compose up -d gemma3n-training

# 2. Verificar que el contenedor está corriendo
docker ps | grep gemma3n-training

# 3. Las dependencias ya están instaladas en el contenedor
# (matplotlib, seaborn, plotly, scikit-learn, scipy, etc.)
```

**Todas las dependencias de evaluación están pre-instaladas en la imagen Docker.**

### Opción 2: Instalación Local (No recomendado)

Si necesitas ejecutar fuera de Docker:

```bash
# Instalar dependencias básicas
pip install -r requirements-file.txt

# Instalar Unsloth (necesario para cargar modelos Gemma3n)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Instalar dependencias de visualización
pip install matplotlib seaborn plotly scikit-learn scipy
```

---

## 📖 Uso Básico

### Uso con Docker (RECOMENDADO)

#### Método 1: Scripts Automatizados (Windows)

```bash
# Evaluación completa
docker-evaluate.bat --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 --save-visuals

# Evaluación rápida (100 muestras)
docker-quick-eval.bat outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1

# Abrir shell interactiva en el contenedor
docker-shell.bat
```

#### Método 2: Comando Docker Directo

```bash
# Evaluación completa
docker exec -it gemma3n-training python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --save-visuals

# Comparación con modelo base
docker exec -it gemma3n-training python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --compare \
    --base-model unsloth/gemma-3n-e2b-it
```

#### Método 3: Shell Interactiva

```bash
# 1. Entrar al contenedor
docker exec -it gemma3n-training /bin/bash

# 2. Dentro del contenedor, ejecutar evaluación
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --save-visuals

# 3. Salir del contenedor
exit
```

### Uso Local (Solo si no usas Docker)

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --test-set prepared_dataset/test_dataset.jsonl \
    --output-dir evaluation_results \
    --save-visuals
```

**Parámetros:**
- `--model-path`: Ruta al modelo entrenado (puede ser local o ID de HuggingFace)
- `--test-set`: Archivo JSONL con el dataset de prueba (por defecto: `prepared_dataset/test_dataset.jsonl`)
- `--output-dir`: Directorio donde guardar resultados (por defecto: `evaluation_results`)
- `--save-visuals`: Guarda comparaciones visuales de predicciones vs ground truth

### 2. Limitar Número de Muestras (Para Pruebas Rápidas)

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --max-samples 100
```

Esto evaluará solo las primeras 100 muestras del dataset de prueba.

### 3. Comparar Modelo Base vs Fine-Tuned

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --compare \
    --base-model unsloth/gemma-3n-e2b-it
```

**Esto generará:**
- Evaluación completa de ambos modelos
- Gráfico de comparación lado a lado
- Tabla con porcentajes de mejora

### 4. Personalizar Umbrales de IoU

```bash
python tools/evaluate_models.py \
    --model-path my_model \
    --iou-thresholds 0.5,0.6,0.7,0.8,0.9
```

---

## 📊 Métricas Calculadas

### 1. **Métricas de Detección (IoU-based)**

Evalúa la capacidad del modelo de localizar objetos correctamente:

- **Precision**: % de detecciones correctas del total de detecciones
- **Recall**: % de objetos detectados del total de objetos reales
- **F1-Score**: Media armónica entre Precision y Recall
- **TP/FP/FN**: True Positives, False Positives, False Negatives

**Criterio de éxito**: Una detección es correcta si:
1. La clase predicha coincide con la ground truth
2. El IoU del bounding box es ≥ 0.5

### 2. **Métricas de Clasificación (Presence-based)**

Evalúa si el modelo identifica las clases presentes (sin considerar localización):

- **Precision**: % de clases predichas que están realmente presentes
- **Recall**: % de clases presentes que fueron detectadas
- **F1-Score**: Media armónica

### 3. **Métricas COCO (mAP)**

Estándar de la industria para detección de objetos:

- **mAP@0.5**: Mean Average Precision con IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision con IoU threshold 0.75
- **mAP@0.5:0.95**: Mean Average Precision promediado sobre IoU thresholds [0.5, 0.55, ..., 0.95]
- **AP por clase**: Average Precision individual para cada categoría

### 4. **Métricas de Rendimiento**

Evalúa la eficiencia del modelo:

- **Mean/Median Latency**: Tiempo promedio/mediano de inferencia (ms)
- **FPS**: Frames por segundo (imágenes/segundo)
- **Parsing Success Rate**: % de respuestas JSON parseadas correctamente

---

## 📁 Estructura de Resultados

Después de ejecutar la evaluación, se generan los siguientes archivos:

```
evaluation_results/
├── main_evaluation_results_20250119_143052.json      # Resultados completos en JSON
├── main_evaluation_summary_20250119_143052.md        # Resumen en Markdown
├── main_evaluation_dashboard_20250119_143052.png    # Dashboard visual completo
├── main_evaluation_report_20250119_143052.html      # Reporte HTML interactivo
├── visual_comparisons/                               # (si --save-visuals)
│   ├── comparison_0000.jpg                           # Primeras 20 imágenes
│   ├── comparison_0001.jpg                           # con predicciones vs GT
│   └── ...
└── model_comparison.png                              # (si --compare)
```

### Contenido del JSON de Resultados

```json
{
  "timestamp": "2025-01-19 14:30:52",
  "model_path": "outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1",
  "num_samples": 2185,
  "detection_metrics": {
    "Plastic_Debris": {
      "precision": 0.45,
      "recall": 0.62,
      "f1": 0.52,
      "tp": 124,
      "fp": 152,
      "fn": 76
    },
    ...
  },
  "detection_overall": {
    "precision": 0.38,
    "recall": 0.51,
    "f1": 0.43
  },
  "classification_metrics": { ... },
  "coco_metrics": {
    "mAP@0.5": 0.32,
    "mAP@0.75": 0.18,
    "mAP@0.5:0.95": 0.21,
    "per_class_ap": {
      "Bottle": 0.28,
      "Can": 0.35,
      ...
    }
  },
  "performance_metrics": {
    "mean_latency_ms": 1243.5,
    "median_latency_ms": 1198.2,
    "fps": 0.80,
    "parsing_success_rate": 0.97
  }
}
```

---

## 📈 Interpretación de Resultados

### ¿Qué es un "Buen" Resultado?

| Métrica | Excelente | Bueno | Aceptable | Pobre |
|---------|-----------|-------|-----------|-------|
| Detection F1 | > 0.80 | 0.60-0.80 | 0.40-0.60 | < 0.40 |
| mAP@0.5 | > 0.70 | 0.50-0.70 | 0.30-0.50 | < 0.30 |
| mAP@0.5:0.95 | > 0.50 | 0.35-0.50 | 0.20-0.35 | < 0.20 |
| FPS | > 5 | 2-5 | 0.5-2 | < 0.5 |

### Comparación: Base vs Fine-tuned

**Modelo Base (Gemma 3n-e2b-it sin fine-tuning):**
- Detection F1: ~0.10
- Classification F1: ~0.55
- mAP@0.5: < 0.05

**Objetivo del Fine-tuning:**
- Detection F1: > 0.40 (4x mejora)
- Classification F1: > 0.70 (30% mejora)
- mAP@0.5: > 0.30 (6x mejora)

---

## 🔍 Casos de Uso Comunes

### Caso 1: Evaluación Rápida Durante el Entrenamiento

Ejecuta evaluaciones periódicas con un subset pequeño:

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/checkpoint-300 \
    --max-samples 50
```

### Caso 2: Evaluación Final Completa

Después del entrenamiento, evalúa el modelo completo:

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --save-visuals
```

### Caso 3: Comparación de Múltiples Checkpoints

Crea un script para evaluar todos los checkpoints:

```bash
for checkpoint in outputs/gemma3n/runs/checkpoint-*; do
    echo "Evaluating $checkpoint"
    python tools/evaluate_models.py \
        --model-path "$checkpoint" \
        --max-samples 100 \
        --output-dir "evaluation_results/$(basename $checkpoint)"
done
```

### Caso 4: Evaluar en Dataset Personalizado

Si tienes un conjunto de prueba diferente:

```bash
python tools/evaluate_models.py \
    --model-path my_model \
    --test-set custom_datasets/underwater_test.jsonl
```

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'unsloth'"

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Error: "CUDA out of memory"

Reduce el tamaño del batch o usa `--max-samples` para evaluar menos imágenes:

```bash
python tools/evaluate_models.py --model-path my_model --max-samples 100
```

### Error: "No module named 'metrics'"

Asegúrate de ejecutar el script desde la raíz del proyecto:

```bash
cd c:/Users/aleja/Desktop/gemma3n
python tools/evaluate_models.py --model-path my_model
```

### Las visualizaciones no se generan

Verifica que matplotlib y seaborn estén instalados:

```bash
pip install matplotlib seaborn
```

---

## 🎯 Mejores Prácticas

### 1. **Evaluación Estratificada**

Primero evalúa en un subset pequeño, luego en el conjunto completo:

```bash
# Prueba rápida
python tools/evaluate_models.py --model-path my_model --max-samples 50

# Evaluación completa
python tools/evaluate_models.py --model-path my_model
```

### 2. **Guarda Siempre las Visualizaciones**

Las comparaciones visuales son cruciales para entender errores:

```bash
python tools/evaluate_models.py --model-path my_model --save-visuals
```

### 3. **Compara con el Baseline**

Siempre evalúa el modelo base para medir la mejora:

```bash
python tools/evaluate_models.py \
    --model-path my_fine_tuned_model \
    --compare \
    --base-model unsloth/gemma-3n-e2b-it
```

### 4. **Documenta los Resultados**

Los archivos JSON y Markdown son perfectos para documentación:

```bash
# Copia los resultados a tu repositorio
cp evaluation_results/main_evaluation_summary_*.md docs/evaluation_report.md
```

---

## 📚 Referencias

### Clases de Basura Marina (8 categorías)

1. **Bottle** - Botellas de plástico/vidrio
2. **Can** - Latas de metal
3. **Fishing_Net** - Redes de pesca
4. **Glove** - Guantes
5. **Mask** - Mascarillas
6. **Metal_Debris** - Desechos metálicos
7. **Plastic_Debris** - Desechos plásticos genéricos
8. **Tire** - Neumáticos

### Dataset de Prueba

- **Archivo**: `prepared_dataset/test_dataset.jsonl`
- **Formato**: JSONL con conversaciones multimodales
- **Tamaño**: ~2,185 imágenes
- **Anotaciones**: Bounding boxes normalizados [0,1] en formato COCO

### Formato de Entrada (JSONL)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "<prompt>"},
        {"type": "image", "image": "/path/to/image.jpg"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "[{\"debris_type\": \"Bottle\", \"material\": \"Plastic\", \"confidence_score\": 1.0, \"bounding_box\": [0.2, 0.3, 0.5, 0.7]}]"
        }
      ]
    }
  ]
}
```

---

## 🤝 Contribuciones

Para añadir nuevas métricas o visualizaciones:

1. **Nuevas métricas**: Edita `tools/metrics.py` y añade una clase heredando de la estructura base
2. **Nuevas visualizaciones**: Edita `tools/visualizations.py` y añade funciones de plotting
3. **Nuevos modos de evaluación**: Edita `tools/evaluate_models.py` en la clase `ModelEvaluator`

---

## 📞 Soporte

Para problemas o preguntas:
- Revisa el código existente en `evaluate_performance.py` (versión original simple)
- Consulta los logs de evaluación en `evaluation_results/`
- Verifica que todas las dependencias estén instaladas correctamente

---

**Última actualización**: 2025-01-19
**Versión**: 1.0.0
**Autor**: OceanGuard AI Team
