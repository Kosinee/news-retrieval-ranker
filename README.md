# 🧠 Semantic News Retrieval System

## 🎯 Цель проекта

Создать систему **семантического поиска финансовых новостей и вопросов** на основе датасета **FiQA-2018**.  
Проект реализует полный цикл работы **retriever-модели (bi-encoder)** с возможностью **непрерывного обучения (Continuous Training)**.

**Основные задачи:**
- Обучить bi-encoder на парах (вопрос, релевантный документ).
- Реализовать векторный поиск с помощью **FAISS**.
- Разработать пайплайн для переобучения модели по мере поступления новых данных.
- Настроить автоматическую оценку качества (IR-метрики) и критерии деплоя.

---

## 🧩 Данные

**Источник:** [BEIR benchmark – FiQA-2018](https://github.com/beir-cellar/beir)

**Описание:**
- Тематика — финансовые новости, вопросы и ответы пользователей.
- Задача — поиск релевантных ответов/документов для пользовательских запросов.

**Файлы:**
- `corpus.jsonl` — корпус документов (`_id`, `title`, `text`);
- `queries.jsonl` — запросы (`_id`, `text`);
- `qrels.tsv` — разметка релевантности (query → doc → rel).

**Где лежат данные и модели (DVC):**
- Raw-данные: `static/data/fiqa`
- Подготовленные данные: `static/data/fiqa_prepared`
- Итоговая модель: `static/artifacts/retriever_fiqa`

**Предобработка:**
- Поле документа: `text_joined = title + " " + text"`;
- Используется `split="train"` для обучения и `split="test"` для оценки;

---

## ⚙️ Целевые метрики

| Категория | Метрика |
|------------|----------|
| **Качество модели** | NDCG@10 ≥ **0.45** |
| | Recall@10 ≥ **0.60** |
| **Сервисные метрики** | Среднее время отклика ≤ **200 мс** |
| | Доля ошибок ≤ **1%** |
| **Ресурсы** | CPU / RAM |

---

## 🧪 План экспериментов

| Этап | Описание | Цель |
|------|-----------|------|
| **1. Базовая модель** | Использовать `all-MiniLM-L6-v2` и обучить на FiQA-2018 | Получить базовую retriever-модель |
| **2. Метрики IR** | Реализовать и протестировать `Recall@k`, `NDCG@k` | Проверить корректность вычислений |
| **4. Deployment** | Внедрить автоматическое обновление индекса и модели при улучшении метрик | Обеспечить непрерывную работу |

---

## 🧰 Запуск проекта

### Установка зависимостей
```bash
# скачиваем зависимости
pip install -r requirements.txt
# восстановление артефактов (если используется DVC)
dvc pull
# прогон пайплайна
dvc repro
# запускаем обучение вручную (если без DVC)
python -m src.training.train_retriever
# сбор метрик
python -m src.training.evaluate_retriever
```

## 🐳 Docker: скоринг по тестовым qrels одной командой

### Сборка образа
```bash
docker build -t ml-app:v1 .
```

### Запуск контейнера
Команда запуска:
```bash
docker run --rm \
  -v "$PWD/static/data/fiqa_prepared:/data" \
  -v "$PWD/static/data:/output" \
  ml-app:v1 \
  --input_path /data \
  --output_path /output/preds.csv
```

### Что делает `src/predict.py`
- Загружает модель из `static/artifacts/retriever_fiqa` (или `--model_path`).
- Берёт первые 100 строк из `qrels/test.tsv`, для каждой строки добавляет негативную пару.
- Считает косинусное сходство и сохраняет CSV `query_id, doc_id, label, score` в `--output_path`, где `label` берётся из qrels (негативы = 0).

## 🧰 TorchServe: онлайн-сервис

### Подготовка артефактов
```bash
bash scripts/build_torchserve_mar.sh
```
Скрипт:
- копирует модель из `static/artifacts/retriever_fiqa` в `torchserve/model_files`;
- сохраняет `model.pt` (state_dict);
- собирает `model-store/mymodel.mar`.
Требуется `torch-model-archiver` (устанавливается вместе с `torchserve`).

### Сборка и запуск контейнера
```bash
docker build -f Dockerfile.torchserve -t mymodel-serve:v1 .
docker run -d -p 8080:8080 -p 8081:8081 mymodel-serve:v1
```

### Пример запроса
```bash
curl -X POST http://localhost:8080/predictions/mymodel -T torchserve/sample_input.json
```
Если включена token‑auth (дефолт TorchServe), используйте токен из контейнера:
```bash
docker exec -it <container_id> cat /home/model-server/key_file.json
curl -X POST http://localhost:8080/predictions/mymodel \
  -H "Authorization: Bearer <INFERENCE_TOKEN>" \
  -T torchserve/sample_input.json
```
Чтобы отключить токены, оставьте `disable_token_auth=true` в `torchserve/config.properties` и пересоберите образ.

### Формат данных
Один запрос:
```json
{"query": "...", "doc": "..."}
```
Батч:
```json
{"instances": [{"query": "...", "doc": "..."}, {"query": "...", "doc": "..."}]}
```
Ответ:
```json
[{"score": 0.123}, {"score": 0.456}]
```

### Конфигурация сервиса
Файл `torchserve/config.properties`:
- `inference_address`, `management_address`, `metrics_address` — адреса API;
- `model_store` — путь к `model-store`;
- `load_models` — какие модели поднимать при старте;
- `default_workers_per_model` — число воркеров;
- `response_timeout` — таймаут ответа (сек).

## 📊 MLflow: локальный трекинг экспериментов

По умолчанию результаты пишутся в локальную папку `mlruns/`.
Чтобы посмотреть UI:
```bash
mlflow ui
```
Откройте `http://127.0.0.1:5000`.

## 📦 DVC: восстановление и воспроизводимость

Полное восстановление версии данных/модели:
```bash
git clone <repo>
cd news-retrieval-ranker
dvc pull
dvc repro
```

### DVC remote через rclone (Google Drive)
В этом проекте DVC remote указывает на локальный путь, смонтированный через rclone:
```bash
rclone mount gdrive: /Users/<user>/mnt/gdrive
```
Проверьте, что в `.dvc/config` указан путь вида:
```
/Users/<user>/mnt/gdrive/news-retrieval-ranker
```
Без активного монтирования `dvc pull` не сможет найти удалённые артефакты.
