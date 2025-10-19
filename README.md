# 🧠 News Retrieval & Ranking System (MLOps Project)

---

## 🎯 Цель проекта

Создать многоступенчатую систему **поиска и рекомендаций новостей**, которая:
1. Генерирует кандидатов (retrieval);
2. Переупорядочивает их по релевантности (reranking);
3. Самообучается в режиме **continuous training** на пользовательских логах.

Основная **бизнес-цель** — увеличить:
- **CTR (Click-Through Rate)** на 10–15 % относительно baseline (BM25);
- **вовлечённость пользователей** за счёт более релевантных рекомендаций.

---

## 📚 Описание данных

### 📦 Источник данных
Используется открытый датасет **[MIND Dataset (Microsoft News)](https://msnews.github.io/)**,  
содержащий реальные пользовательские логи новостного сервиса **Microsoft News**.

**Период сбора:** 6 недель (октябрь–ноябрь 2019)  
**Объём данных:**
- 1 000 000 пользователей;
- 161 013 новостных статей;
- 15 777 377 импрессий (показов новостей);
- 24 155 470 кликов.

---

### 🧾 Формат данных

| Поле | Описание |
|------|-----------|
| `user_id` | Анонимизированный идентификатор пользователя |
| `time` | Время импрессии (показа блока новостей) |
| `click_history` | Список ID новостей, ранее кликнутых пользователем |
| `impression_log` | Список `(news_id, label)`, где label = 1 (click) или 0 (no click) |
| `news_id` | Уникальный ID новости |
| `title`, `abstract`, `body` | Текстовое описание статьи |
| `category` | Тематическая категория новости |
| `entities` | Сущности, извлечённые из текста и связанных статей (WikiData) |

---

### 📊 Статистика
| Параметр | Значение |
|-----------|-----------|
| Средняя длина заголовка | 11.5 слов |
| Средняя длина аннотации | 43 слова |
| Средняя длина текста | 585 слов |
| Кол-во категорий | 20 |
| Среднее число кликов на пользователя | 24 |

📘 Основная особенность: данные содержат **и текстовую, и поведенческую информацию**,  
что позволяет обучать как семантические модели (retrievers), так и поведенческие (LTR).

---

## 🧪 Описание эксперимента

### 🧠 Гипотеза

> Использование многоступенчатой архитектуры  
> (retriever → reranker) с семантическим и поведенческим обучением  
> улучшит **NDCG@10** и повысит **CTR** на 10–15 %  
> по сравнению с baseline (BM25).

---

### ⚙️ Архитектура системы


---

### 🧩 Компоненты эксперимента

#### 1️⃣ Candidate Generators (Retriever)
| Модель | Тип | Цель | Лосс | Метрики |
|--------|------|------|------|----------|
| **BM25** | Keyword | базовая полнотекстовая модель | — | Recall@K |
| **DSSM_sem** | Bi-encoder | семантическая близость | `InfoNCE` | Recall@K, MRR@10 |
| **DSSM_pref** | Bi-encoder | поведенческая близость (по кликам) | `TripletLoss` | Recall@K, MRR@10 |

#### 2️⃣ Ранжировщик (Ranker)
| Модель | Тип | Лосс | Цель | Метрики |
|--------|------|------|------|----------|
| **LightGBM LambdaMART** | Listwise Ranker | `LambdaMART Loss` | оптимизация NDCG | NDCG@10, MAP, MRR |

**Фичи для LTR:**
- скоры retrievers (BM25, DSSM_sem, DSSM_pref);
- CTR и статистика по новостям (поведенческие);
- категория, источник (категориальные);
- интеракционные признаки (например, `score_sem * score_pref`).

---

### 📊 Метрики

| Уровень | Метрики | Интерпретация | Целевое значение |
|----------|----------|----------------|------------------|
| **Retriever** | Recall@100, MRR@10 | полнота и скорость поиска | Recall ≥ 0.9 |
| **Ranker** | NDCG@10, MAP | качество порядка в выдаче | NDCG@10 ≈ 0.55 |
| **Бизнес** | CTR uplift | вовлечённость | +10–15 % |
| **Сервисные** | Latency P95 | производительность | ≤ 200 мс |

---

### 🔬 Дизайн эксперимента

| Этап | Цель | Модель / Метод | Метрики |
|------|------|----------------|----------|
| **1. Baseline** | Оценить BM25 | BM25 | Recall@100, NDCG@10 |
| **2. Семантический retriever** | Проверить DSSM_sem | DSSM + InfoNCE | Recall@100, MRR@10 |
| **3. Поведенческий retriever** | Учесть предпочтения | DSSM_pref + TripletLoss | Recall@100, MRR@10 |
| **4. Ensemble retrieval** | Объединить retrievers | Union(top-K) | Recall@100 |
| **5. Ranker (LTR)** | Упорядочить кандидатов | LightGBM LambdaMART | NDCG@10, MAP |
| **6. A/B Test** | Проверить влияние на бизнес-метрики | Production pipeline | CTR uplift |

---

### ⚖️ A/B-тестирование

| Группа | Модель | Трафик | Метрики |
|---------|---------|--------|----------|
| **Control** | BM25 baseline | 50 % | CTR₀, NDCG₀ |
| **Treatment** | DSSM + LambdaMART | 50 % | CTR₁, NDCG₁ |

**Критерий успеха:**
\[
CTR_{uplift} = \frac{CTR_1 - CTR_0}{CTR_0} \ge 10\%, \quad p < 0.05
\]

---

### 🧮 Offline-процедура
1. Разделить MIND на train / validation / test (по неделям).  
2. Обучить retrievers на train, ranker — на validation.  
3. На test — измерить метрики (Recall, NDCG, CTR).  
4. Сравнить с baseline BM25.  
5. Зафиксировать улучшения и провести анализ чувствительности (по K и фичам).

---

### 📈 Ожидаемые результаты

| Модель | Recall@100 | NDCG@10 | CTR uplift |
|---------|-------------|----------|-------------|
| BM25 | 0.85 | 0.48 | — |
| DSSM_sem | 0.89 | 0.51 | — |
| DSSM_pref | 0.92 | 0.53 | — |
| DSSM + LambdaMART | 0.94 | 0.56 | +12 % |

---

### ✅ Критерии успешности
- **Технические:** NDCG@10 ≥ 0.55, Recall@100 ≥ 0.9  
- **Бизнес:** CTR uplift ≥ +10 %, p-value < 0.05  
- **Инфраструктурные:** latency ≤ 200 мс, retraining ≤ 2 ч  

---

## 🔁 Continuous Training Pipeline

1. **Сбор логов** (новые импрессии и клики).  
2. **Обновление фичей** (feature engineering job).  
3. **Переобучение retrievers и ranker.**  
4. **Оценка метрик** (auto-eval job).  
5. **Автоматический деплой** при улучшении NDCG@10 > threshold.  

---

## 🧩 Используемые технологии
- **PyTorch** — DSSM retrievers  
- **LightGBM** — LambdaMART ranker  
- **MLflow / Airflow** — Continuous training & evaluation  
- **Docker / FastAPI** — inference-сервис  
- **Pandas / PySpark** — feature pipelines  

---

## 📘 Литература
- *Burges et al., "From RankNet to LambdaMART: The LambdaRank Framework"*  
- *Wu et al., "Neural News Recommendation with Multi-Head Self-Attention (NRMS)"*  
- *Microsoft MIND Dataset (2019)*  
- *LightGBM Documentation – Lambdarank Objective*  
- *Recommenders by Microsoft (GitHub)*  

---

✍️ **Автор:** Твоё имя  
📧 your.email@example.com  
💼 [GitHub / LinkedIn / Kaggle]

---
