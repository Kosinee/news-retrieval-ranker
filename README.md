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
- +10-15 % CTR uplift относительно baseline (BM25)

---

## 🧩 Компоненты

### 1️⃣ Candidate Generators
| Модель | Тип | Цель | Лосс | Основные метрики |
|--------|------|------|------|------------------|
| **BM25** | Keyword | полнотекстовая близость | — | Recall@K |
| **DSSM_sem** | Bi-encoder | семантическая близость | `InfoNCE` | Recall@K |
| **DSSM_pref** | Bi-encoder | предпочтения по кликам | `TripletLoss` | Recall@K |

---

### 2️⃣ Reranker / Ranker
| Модель | Тип | Лосс | Оптимизирует | Метрики |
|--------|------|------|---------------|----------|
| **LightGBM LambdaMART** | Listwise Ranker | `LambdaMART Loss` | NDCG | NDCG@10, MRR, MAP |

**Основные признаки:**
- скоры от BM25, DSSM_sem, DSSM_pref;  
- CTR и статистика документов (поведенческие фичи);  
- категории / источники (категориальные);  

---

## 📊 Метрики

| Уровень | Метрики | Интерпретация | Целевые значения |
|----------|----------|---------------|------------------|
| **Retriever** | Recall@100, MRR@10 | полнота и скорость нахождения релевантных | Recall≥0.9 |
| **Ranker** | NDCG@10, MRR@10 | качество порядка в выдаче | NDCG@10≈0.55 |
| **Бизнес** | CTR | вовлечённость пользователей | +10–15 % CTR uplift |
| **Сервисные** | Latency | стабильность пайплайна | ≤ 200 мс |

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
