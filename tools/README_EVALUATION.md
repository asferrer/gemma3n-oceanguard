# 🎯 Sistema de Evaluación de Modelos Gemma3n

Sistema completo de evaluación para modelos Gemma3n de detección de basura marina con métricas avanzadas y visualizaciones.

## 📂 Archivos

```
tools/
├── metrics.py              # Clases de cálculo de métricas
├── visualizations.py       # Generación de gráficos y reportes
├── evaluate_models.py      # Script principal de evaluación
├── test_metrics.py         # Tests de validación
└── README_EVALUATION.md    # Esta guía
```

## 🚀 Quick Start

### 1. Verificar Instalación

```bash
# Probar que los módulos funcionan
python tools/test_metrics.py
```

Deberías ver:
```
✅ ALL TESTS PASSED!
Los módulos de métricas funcionan correctamente.
```

### 2. Evaluar tu Modelo

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1
```

### 3. Revisar Resultados

Los resultados se guardan en `evaluation_results/`:
- 📊 **Dashboard PNG**: Gráfico completo con todas las métricas
- 📄 **Resumen Markdown**: Tabla de resultados
- 🌐 **Reporte HTML**: Dashboard interactivo
- 💾 **JSON**: Datos completos para análisis posterior

## 📊 Métricas Disponibles

### Detection Metrics (IoU-based)
- ✅ Precision, Recall, F1 por clase
- ✅ True Positives, False Positives, False Negatives
- ✅ Matching basado en IoU threshold

### Classification Metrics
- ✅ Evaluación de presencia de clases
- ✅ Precision, Recall, F1 por clase

### COCO Metrics
- ✅ mAP@0.5
- ✅ mAP@0.75
- ✅ mAP@0.5:0.95
- ✅ Average Precision por clase

### Performance Metrics
- ✅ Latencia (mean, median, min, max)
- ✅ Throughput (FPS)
- ✅ Tasa de parsing JSON exitoso

## 🎨 Visualizaciones

### Dashboard Completo
![Dashboard Example](../docs/evaluation_dashboard_example.png)

Incluye:
- Gráficos de barras por métrica y clase
- Distribución de clases en el dataset
- Métricas COCO (mAP)
- AP por clase
- Métricas de rendimiento
- Tabla resumen

### Comparación de Modelos
![Comparison Example](../docs/model_comparison_example.png)

Compara lado a lado:
- Modelo base vs fine-tuned
- Porcentaje de mejora
- Todas las métricas principales

### Visualizaciones de Detecciones
![Detection Example](../docs/detection_comparison_example.png)

Imagen lado a lado:
- Predicciones del modelo (cyan)
- Ground truth (verde)

## 💡 Ejemplos de Uso

### Ejemplo 1: Evaluación Estándar

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --test-set prepared_dataset/test_dataset.jsonl \
    --output-dir evaluation_results
```

**Salida:**
```
🔬 Starting Comprehensive Model Evaluation
📖 Loading test set from: prepared_dataset/test_dataset.jsonl
✅ Loaded 2185 test samples
🔍 Processing 2185 test samples...
  Progress: 50/2185 samples
  ...
✅ Evaluation complete!

📊 EVALUATION SUMMARY
Detection F1:      0.4321
Classification F1: 0.7123
mAP@0.5:          0.3245
mAP@0.5:0.95:     0.2156
Mean Latency:      1234.56 ms
FPS:               0.81
```

### Ejemplo 2: Evaluación Rápida (100 muestras)

```bash
python tools/evaluate_models.py \
    --model-path my_model \
    --max-samples 100
```

**Tiempo estimado:** ~2-3 minutos

### Ejemplo 3: Comparación con Base Model

```bash
python tools/evaluate_models.py \
    --model-path outputs/gemma3n/runs/gemma3n_e4b_waste_peft_v1 \
    --compare \
    --base-model unsloth/gemma-3n-e2b-it
```

**Salida:**
```
🔍 MODEL COMPARISON
Metric               |    Base Model |    Fine-tuned |    Improvement
----------------------------------------------------------------------
detection_f1         |        0.1000 |        0.4321 |       +332.10%
classification_f1    |        0.5500 |        0.7123 |        +29.51%
mAP@0.5              |        0.0452 |        0.3245 |       +617.92%
mAP@0.5:0.95         |        0.0234 |        0.2156 |       +821.37%
```

### Ejemplo 4: Con Visualizaciones

```bash
python tools/evaluate_models.py \
    --model-path my_model \
    --save-visuals
```

Guarda las primeras 20 comparaciones visuales en `evaluation_results/visual_comparisons/`

### Ejemplo 5: Umbrales IoU Personalizados

```bash
python tools/evaluate_models.py \
    --model-path my_model \
    --iou-thresholds 0.3,0.4,0.5,0.6,0.7
```

## 📈 Interpretación de Resultados

### ¿Qué significan las métricas?

#### Detection F1 Score
- **> 0.80**: Excelente - Listo para producción
- **0.60-0.80**: Bueno - Usable con supervisión
- **0.40-0.60**: Aceptable - Necesita mejoras
- **< 0.40**: Pobre - Requiere más entrenamiento

#### mAP@0.5
- **> 0.70**: Excelente
- **0.50-0.70**: Bueno
- **0.30-0.50**: Aceptable
- **< 0.30**: Pobre

#### Classification F1
- **> 0.80**: Excelente en identificar clases presentes
- **0.60-0.80**: Bueno
- **< 0.60**: Necesita mejora

### Mejora Esperada con Fine-tuning

| Métrica | Base Model | Target (Fine-tuned) | Mejora |
|---------|------------|---------------------|--------|
| Detection F1 | ~0.10 | > 0.40 | 4x |
| mAP@0.5 | ~0.05 | > 0.30 | 6x |
| Classification F1 | ~0.55 | > 0.70 | 30% |

## 🔧 Personalización

### Añadir Nuevas Métricas

Edita `tools/metrics.py`:

```python
class CustomMetrics:
    def __init__(self):
        self.custom_stat = 0

    def update(self, predictions, ground_truths):
        # Tu lógica aquí
        pass

    def get_metrics(self):
        return {'custom_metric': self.custom_stat}
```

Luego integra en `evaluate_models.py`:

```python
from metrics import CustomMetrics

# En __init__ de ModelEvaluator:
self.custom_metrics = CustomMetrics()

# En evaluate():
self.custom_metrics.update(predictions, ground_truths)
results['custom_metrics'] = self.custom_metrics.get_metrics()
```

### Añadir Nuevas Visualizaciones

Edita `tools/visualizations.py`:

```python
def plot_custom_chart(data, output_path):
    """Tu función de visualización."""
    fig, ax = plt.subplots(figsize=(12, 8))
    # Tu código aquí
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

## 🐛 Troubleshooting

### Error: "No module named 'unsloth'"
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Error: "CUDA out of memory"
```bash
# Evalúa menos muestras
python tools/evaluate_models.py --model-path my_model --max-samples 50
```

### Error: "File not found: test_dataset.jsonl"
```bash
# Verifica que el archivo existe
ls -la prepared_dataset/test_dataset.jsonl

# O especifica la ruta correcta
python tools/evaluate_models.py \
    --model-path my_model \
    --test-set /path/to/your/test_set.jsonl
```

### Advertencia: "Parsing failed for X samples"
Esto es normal si el modelo a veces genera JSON malformado. La tasa de éxito de parsing se reporta en las métricas de rendimiento.

## 📊 Formato del Dataset de Test

El script espera un archivo JSONL donde cada línea es:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this underwater image..."},
        {"type": "image", "image": "/absolute/path/to/image.jpg"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "[{\"debris_type\": \"Bottle\", \"material\": \"Plastic\", \"confidence_score\": 1.0, \"bounding_box\": [0.1, 0.2, 0.3, 0.4]}]"
        }
      ]
    }
  ]
}
```

## 🎓 Best Practices

1. **Siempre compara con el baseline:**
   ```bash
   --compare --base-model unsloth/gemma-3n-e2b-it
   ```

2. **Guarda visualizaciones para análisis:**
   ```bash
   --save-visuals
   ```

3. **Evalúa checkpoints intermedios:**
   ```bash
   for ckpt in outputs/gemma3n/runs/checkpoint-*; do
       python tools/evaluate_models.py --model-path "$ckpt" --max-samples 100
   done
   ```

4. **Documenta tus experimentos:**
   - Guarda los JSON de resultados
   - Anota cambios en hyperparámetros
   - Compara múltiples versiones

## 📞 Soporte

Para más información, consulta:
- 📖 [Guía de Evaluación Completa](../EVALUATION_GUIDE.md)
- 🔬 [Script Original](../evaluate_performance.py)
- 💻 [Código de Entrenamiento](train.py)

---

**Versión:** 1.0.0
**Última actualización:** 2025-01-19
**Mantenedor:** OceanGuard AI Team
