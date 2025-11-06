---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Business Expense - Car Insurance Deductible For Accident That Occurred
    During a Business Trip
  sentences:
  - 'The IRS Guidance pertaining to the subject.  In general the best I can say is
    your business expense may be deductible.  But it depends on the circumstances
    and what it is you want to deduct. Travel Taxpayers who travel away from home
    on business may deduct related   expenses, including the cost of reaching their
    destination, the cost   of lodging and meals and other ordinary and necessary
    expenses.   Taxpayers are considered “traveling away from home” if their duties   require
    them to be away from home substantially longer than an   ordinary day’s work and
    they need to sleep or rest to meet the demands   of their work. The actual cost
    of meals and incidental expenses may be   deducted or the taxpayer may use a standard
    meal allowance and reduced   record keeping requirements. Regardless of the method
    used, meal   deductions are generally limited to 50 percent as stated earlier.    Only
    actual costs for lodging may be claimed as an expense and   receipts must be kept
    for documentation. Expenses must be reasonable   and appropriate; deductions for
    extravagant expenses are not   allowable. More information is available in Publication
    463, Travel,   Entertainment, Gift, and Car Expenses. Entertainment Expenses for
    entertaining clients, customers or employees may be   deducted if they are both
    ordinary and necessary and meet one of the   following tests: Directly-related
    test: The main purpose of the entertainment activity is the conduct of business,
    business was actually conducted   during the activity and the taxpayer had more
    than a general   expectation of getting income or some other specific business
    benefit   at some future time.   Associated test: The entertainment was associated
    with the active conduct of the taxpayer’s trade or business and occurred directly   before
    or after a substantial business discussion. Publication 463 provides more extensive
    explanation of these tests as   well as other limitations and requirements for
    deducting entertainment   expenses. Gifts Taxpayers may deduct some or all of
    the cost of gifts given in the   course of their trade or business. In general,
    the deduction is   limited to $25 for gifts given directly or indirectly to any
    one   person during the tax year. More discussion of the rules and   limitations
    can be found in Publication 463. If your LLC reimburses you for expenses outside
    of this guidance it should be treated as Income for tax purposes. Edit for Meal
    Expenses: Amount of standard meal allowance.   The standard meal allowance is   the
    federal M&IE rate. For travel in 2010, the rate for most small   localities in
    the United States is $46 a day. Source IRS P463 Alternately you could reimburse
    at a per diem rate'
  - As a general rule, you must choose between a mileage deduction or an actual expenses
    deduction.  The idea is that the mileage deduction is supposed to cover all costs
    of using the car.  Exceptions include parking fees and tolls, which can be deducted
    separately under either method.  You explicitly cannot deduct insurance costs
    if you claim a mileage deduction.   Separately, you probably won't be able to
    deduct the deductible for your car as a casualty loss.  You first subtract $100
    from the deductible and then divide it by your Adjusted Gross Income (AGI) from
    your tax return.  If your deductible is over 10% of your AGI, you can deduct it.   Note
    that even with a $1500 deductible, you won't be able to deduct anything if you
    made more than $14,000 for the year.  For most people, the insurance deductible
    just isn't large enough relative to income to be tax deductible.   Source
  - I don't believe Saturday is a business day either. When I deposit a check at a
    bank's drive-in after 4pm Friday, the receipt tells me it will credit as if I
    deposited on Monday.  If a business' computer doesn't adjust their billing to
    have a weekday due date, they are supposed to accept the payment on the next business
    day, else, as you discovered, a Sunday due date is really the prior Friday. In
    which case they may be running afoul of the rules that require X number of days
    from the time they mail a bill to the time it's due.  The flip side to all of
    this, is to pick and choose your battles in life. Just pay the bill 2 days early.
    The interest on a few hundred dollars is a few cents per week. You save that by
    not using a stamp, just charge it on their site on the Friday. Keep in mind, you
    can be right, but their computer still dings you. So you call and spend your valuable
    time when ever the due date is over a weekend, getting an agent to reverse the
    late fee. The cost of 'right' is wasting ten minutes, which is worth far more
    than just avoiding the issue altogether.  But - if you are in the US (you didn't
    give your country), we have regulations for everything. HR 627, aka The CARD act
    of 2009, offers - ‘‘(2) WEEKEND OR HOLIDAY DUE DATES.—If the payment due date
    for a   credit card account under an open end consumer credit plan is a day on   which
    the creditor does not receive or accept payments by mail   (including weekends
    and holidays), the creditor may not treat a   payment received on the next business
    day as late for any purpose.’’. So, if you really want to pursue this, you have
    the power of our illustrious congress on your side.
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Business Expense - Car Insurance Deductible For Accident That Occurred During a Business Trip',
    "As a general rule, you must choose between a mileage deduction or an actual expenses deduction.  The idea is that the mileage deduction is supposed to cover all costs of using the car.  Exceptions include parking fees and tolls, which can be deducted separately under either method.  You explicitly cannot deduct insurance costs if you claim a mileage deduction.   Separately, you probably won't be able to deduct the deductible for your car as a casualty loss.  You first subtract $100 from the deductible and then divide it by your Adjusted Gross Income (AGI) from your tax return.  If your deductible is over 10% of your AGI, you can deduct it.   Note that even with a $1500 deductible, you won't be able to deduct anything if you made more than $14,000 for the year.  For most people, the insurance deductible just isn't large enough relative to income to be tax deductible.   Source",
    "I don't believe Saturday is a business day either. When I deposit a check at a bank's drive-in after 4pm Friday, the receipt tells me it will credit as if I deposited on Monday.  If a business' computer doesn't adjust their billing to have a weekday due date, they are supposed to accept the payment on the next business day, else, as you discovered, a Sunday due date is really the prior Friday. In which case they may be running afoul of the rules that require X number of days from the time they mail a bill to the time it's due.  The flip side to all of this, is to pick and choose your battles in life. Just pay the bill 2 days early. The interest on a few hundred dollars is a few cents per week. You save that by not using a stamp, just charge it on their site on the Friday. Keep in mind, you can be right, but their computer still dings you. So you call and spend your valuable time when ever the due date is over a weekend, getting an agent to reverse the late fee. The cost of 'right' is wasting ten minutes, which is worth far more than just avoiding the issue altogether.  But - if you are in the US (you didn't give your country), we have regulations for everything. HR 627, aka The CARD act of 2009, offers - ‘‘(2) WEEKEND OR HOLIDAY DUE DATES.—If the payment due date for a   credit card account under an open end consumer credit plan is a day on   which the creditor does not receive or accept payments by mail   (including weekends and holidays), the creditor may not treat a   payment received on the next business day as late for any purpose.’’. So, if you really want to pursue this, you have the power of our illustrious congress on your side.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 8 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                           |
  |:--------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                               |
  | details | <ul><li>min: 6 tokens</li><li>mean: 12.5 tokens</li><li>max: 19 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 113.62 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
  |:-----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Business Expense - Car Insurance Deductible For Accident That Occurred During a Business Trip</code> | <code>As a general rule, you must choose between a mileage deduction or an actual expenses deduction.  The idea is that the mileage deduction is supposed to cover all costs of using the car.  Exceptions include parking fees and tolls, which can be deducted separately under either method.  You explicitly cannot deduct insurance costs if you claim a mileage deduction.   Separately, you probably won't be able to deduct the deductible for your car as a casualty loss.  You first subtract $100 from the deductible and then divide it by your Adjusted Gross Income (AGI) from your tax return.  If your deductible is over 10% of your AGI, you can deduct it.   Note that even with a $1500 deductible, you won't be able to deduct anything if you made more than $14,000 for the year.  For most people, the insurance deductible just isn't large enough relative to income to be tax deductible.   Source</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | <code>Hobby vs. Business</code>                                                                            | <code>Miscellaneous income -- same category used for hobbies.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
  | <code>“Business day” and “due date” for bills</code>                                                       | <code>You definitely have an argument for getting them to reverse the late fee, especially if it hasn't happened very often. (If you are late every month they may be less likely to forgive.) As for why this happens, it's not actually about business days, but instead it's based on when they know that you paid. In general, there are 2 ways for a company to mark a bill as paid: Late Fees: Some systems automatically assign late fees at the start of the day after the due date if money has not been received. In your case, if your bill was due on the 24th, the late fee was probably assessed at midnight of the 25th, and the payment arrived after that during the day of the 25th. You may have been able to initiate the payment on the company's website at 11:59pm on the 24th and not have received a late fee (or whatever their cutoff time is). Suggestion: as a rule of thumb, for utility bills whose due date and amount can vary slightly from month to month, you're usually better off setting up your payments on the company website to pull from your bank account, instead of setting up your bank account to push the payment to the company. This will ensure that you always get the bill paid on time and for the correct amount. If you still would rather push the payment from your bank account, then consider setting up the payment to arrive about 5 days early, to account for holidays and weekends.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `num_train_epochs`: 1
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.4
- Sentence Transformers: 3.0.1
- Transformers: 4.43.3
- PyTorch: 2.3.1+cu121
- Accelerate: 1.11.0
- Datasets: 4.4.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply}, 
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->