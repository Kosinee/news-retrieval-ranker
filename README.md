# 🧠 News Retrieval & Ranking System (MLOps Project)

## 📌 Описание проекта
Цель — построить многоступенчатую систему **поиска и рекомендаций новостей**,  
которая умеет:
1. генерировать кандидатов (retrieval);
2. их переупорядочивать (reranking);
3. обучаться в режиме **continuous training** по пользовательским логам.

Основа данных — **[MIND Dataset (Microsoft News)](https://msnews.github.io/)**,  
в котором содержатся описания новостей и пользовательские клики (impressions).

---

## 💼 Бизнес-цель
Увеличить **CTR (Click-Through Rate)** и **вовлечённость пользователей**  
за счёт более релевантной выдачи новостей на главной странице.  
Цель:
- +10-15 % CTR uplift относительно baseline (BM25),
- уменьшить неинтересные рекомендации в top-5.

---

## 🏗️ Архитектура пайплайна

           ┌──────────────────────┐
           │  User Query / Context│
           └──────────┬───────────┘
                      │
    ┌─────────────────▼─────────────────┐
    │          Candidate Generation     │
    │───────────────────────────────────│
    │  • BM25 (keyword retriever)       │
    │  • DSSM_sem (semantic bi-encoder) │
    │  • DSSM_pref (behavioral bi-encoder)│
    └───────────────┬───────────────────┘
                    │
    ┌───────────────▼───────────────────┐
    │          Ranking Layer (LTR)      │
    │───────────────────────────────────│
    │  LightGBM LambdaMART Ranker       │
    │  Фичи:                            │
    │   – скоры retrievers              │
    │   – поведенческие (CTR_doc, freq) │
    │   – категориальные / бизнес       │
    │   – интеракционные                │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────▼───────────────────┐
    │        Continuous Training Loop   │
    │───────────────────────────────────│
    │  1. сбор новых логов              │
    │  2. подготовка данных             │
    │  3. переобучение retrievers/LTR   │
    │  4. деплой обновлённых моделей    │
    └───────────────────────────────────┘

---

## 🧩 Компоненты

### 1️⃣ Candidate Generators
| Модель | Тип | Цель | Лосс | Основные метрики |
|--------|------|------|------|------------------|
| **BM25** | Keyword | базовый baseline | — | Recall@100 |
| **DSSM_sem** | Bi-encoder | семантическая близость | `InfoNCE` | Recall@K, MRR |
| **DSSM_pref** | Bi-encoder | предпочтения по кликам | `TripletLoss` | Recall@K, NDCG@K |

---

### 2️⃣ Reranker / Ranker
| Модель | Тип | Лосс | Оптимизирует | Метрики |
|--------|------|------|---------------|----------|
| **LightGBM LambdaMART** | Listwise Ranker | `LambdaMART Loss` (взвешенный pairwise-логистический) | NDCG | NDCG@10, MRR |

**Основные признаки:**
- скоры от BM25, DSSM_sem, DSSM_pref;  
- CTR и статистика документов (поведенческие фичи);  
- категории / источники (категориальные);  
- интеракции: `(query_category == doc_category)`, `(score_sem × score_pref)` и др.

---

## 🧮 Функции потерь

| Этап | Лосс | Формула | Что оптимизирует |
|------|------|----------|------------------|
| Retriever | **InfoNCE** | ![formula](https://latex.codecogs.com/svg.image?L=-\log\frac{e^{sim(q,d^+)/\tau}}{\sum_j e^{sim(q,d_j)/\tau)}) | максимизация сходства релевантных пар |
| Retriever | **TripletLoss** | ![formula](https://latex.codecogs.com/svg.image?L=\max(0,m-sim(q,d^%2B)%2Bsim(q,d^-))) | margin между релевантным и нерелевантным |
| Ranker | **LambdaMART** | ![formula](https://latex.codecogs.com/svg.image?L=\sum_{i<j}|\Delta%20NDCG_{ij}|\log(1&plus;e^{-(s_i-s_j)}) ) | улучшение порядка (NDCG) |

---

## 📊 Метрики

| Уровень | Метрики | Интерпретация | Целевые значения |
|----------|----------|---------------|------------------|
| **Retriever** | Recall@100, MRR@10 | полнота и скорость нахождения релевантных | Recall≥0.9, MRR≈0.35 |
| **Ranker** | NDCG@10, MRR@10 | качество порядка в выдаче | NDCG@10≈0.55 |
| **Бизнес** | CTR, Dwell Time | вовлечённость пользователей | +10–15 % CTR uplift |
| **Сервисные** | Latency P95, Drift | стабильность пайплайна | ≤ 200 мс, ΔNDCG < 5 % |

---

## 🔁 Continuous Training Workflow

1. **Сбор новых логов** из продакшн-потока (импрессии, клики).  
2. **Data Prep Job** — обновление таблиц features.  
3. **Retraining retrievers** (DSSM_sem / DSSM_pref) по свежим логам.  
4. **Retraining ranker** (LightGBM LambdaMART).  
5. **Evaluation** на hold-out неделе.  
6. **Auto-deploy** модели при улучшении NDCG@10 > threshold.  

---

## ⚙️ Пример обучения ранжировщика (LightGBM)

```python
import lightgbm as lgb

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9
}

train = lgb.Dataset(X_train, label=y_train, group=group_train)
valid = lgb.Dataset(X_valid, label=y_valid, group=group_valid)

model = lgb.train(params, train, valid_sets=[valid],
                  num_boost_round=200, early_stopping_rounds=30)
