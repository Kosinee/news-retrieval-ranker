# 🧠 News Retrieval (BEIR TREC-News) with Continuous Training

## 🎯 Цель
Семантический поиск новостей с непрерывным обучением (CT): retriever (bi-encoder) → FAISS → оффлайн метрики → blue/green обновление индекса и модели без простоя.

## Данные
- **BEIR / TREC-News**: `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`.
- Текст документа = `title + text` → поле `text_joined`.
- Псевдо-временной поток: документы подаются батчами (или по `metadata.date`, если есть).

## Метрики
- **Retriever**: Recall@10/100, NDCG@10/100.
- **Сервис**: Latency P95 ≤ 200 мс, Error rate ≤ 1%.
- **CT-порог деплоя**: ΔNDCG@10 ≥ 0.002 против предыдущей версии.

## Запуск
```bash
pip install -r requirements.txt

# Continuous Training
python -m src.training.continuous_train --config config/ct.yaml
# артефакты:
#  - модели: artifacts/retriever/v{N}
#  - индексы: artifacts/index/index_v{N}, симлинк current -> активная версия
#  - метрики: artifacts/ct_metrics.json
