---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:5498
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Is there a country that uses the term “dollar” for currency without
    also using “cents” as fractional monetary units?
  sentences:
  - 'The S&P 500 is a stock market index, which is a list of 500 stocks from the largest
    companies in America. You could open a brokerage account with a broker and buy
    shares in each of these companies, but the easiest, least expensive way to invest
    in all these stocks is to invest in an S&P 500 index mutual fund. Inside an index
    mutual fund, your money will be pooled together with everyone else in the fund
    to purchase all the stocks in the index.  These types of funds are very low expense
    compared to managed mutual funds.  Most mutual fund companies have an S&P 500
    index fund; two examples are Vanguard and Fidelity.  The minimum investment in
    most of these mutual funds is low enough that you will be able to open an account
    with your $4000. Something you need to keep in mind, however: investing in any
    stock mutual fund is not non-risk.  It''s not even low-risk, really.  It is very
    possible to lose money by investing in the stock market.  An S&P 500 index fund
    is diversified in the sense that you have money in lots of different stocks, but
    it is also not diversified, in a sense, because it is all in large cap American
    stocks.  Before investing in the stock market, you should have a goal for the
    money you are investing.  If you are investing for something several years away,
    an index fund can be a good place to invest, but if you will need this money within
    the next few years, the stock market might be too risky for you.'
  - '"Going through the list of economies that currently use the dollar, all of them
    list cents as a fractional unit. In Hong Kong and Taiwan, the 1/100 fractional
    unit is still called a cent, but it''s no longer in circulation in coin form and
    only finds use in financial markets or electronic payments.  In countries like
    Malaysia, the word ""sen"" is used as the translation of the word ""cent"", even
    though the word for the actual currency, ""ringgit"", isn''t a translation of
    the word ""dollar"".  A similar situation occurs in Panama. The local currency
    is called the balboa, and it''s priced on par (1:1) with the US dollar. US banknotes
    are also accepted as legal tender, and Panamanians sometimes use the terms balboa/dollar
    interchangeably. The 1/100 subdivision of the balboa is the centésimo, which is
    merely a translation of cent. Like Malaysia, the fractional unit is called ""cent""
    (or a translation) but the main unit isn''t merely a translation of the word ""dollar.""
    On a historical note, the Spanish Dollar was subdivided into 8 reales in order
    to match the German thaler (the word that forms the basis for the English word
    ""dollar"")."'
  - '"Moneydance is a commercial application that is cross-platform. Written in Java,
    they run and are supported on Windows, Mac and Linux. They integrate with many
    financial institutions and for those that it cannot, you can import a locally
    downloaded file. I have used it for several years on my Mac, but have no company
    affiliation. I''m not sure if by saying ""Unix"" software you meant FOSS of some
    kind, but good luck in any case."'
- source_sentence: Borrowing 100k and paying it to someone then declaring bankruptcy
  sentences:
  - '"If the wording is ""within 10 days"" then its 10 days. Calendar days. Otherwise
    they would put ""10 business days"", for example. Usually, if you need to do something
    within 10 days from today, the first day to count is today. I would expect ""within""
    to mean that you can fund in any of the days up to the 10th. But that''s me, trying
    to read English as English. Why don''t you call the bank and ask them?"'
  - '"Basically, these guys break all your eggs then try to make an omelet.  Your
    lender(s) must really believe that you have no ability to pay before they''ll
    settle, which generally entails not paying them until your creditworthiness is
    in the tank.  Bankruptcy laws exist for a reason.  If your credit is in the tank,
    you can''t make your payments and you''re shopping to settle your debts, it''s
    not likely a bankruptcy would worsen your situation; in fact, quite the opposite.  But,
    people have hugely negative feelings toward bankruptcy and don''t want to be called
    a ""deadbeat"", these services prey on those people."'
  - This sounds like a crazy idea, but in reality people don't make the wisest decisions
    when considering bankruptcy in Australia.  My suggestion would be to get some
    advice from an insolvency specialist.
- source_sentence: Can my U.S. company do work for a foreign company and get wire
    transfers to my personal account?
  sentences:
  - 'There are a few ways to look at this question. Assumptions. Per the original
    post''s assumptions, this answer: In other words, if the owner paid the mortgage
    on its original schedule, the deal could boil down to a $ 40,000 up-front payment,
    in exchange for $ 200,000 of equity after 30 years.  Or the deal could boil down
    to a $ 40,000 up-front payment, in exchange for a $ 810.70 monthly payment starting
    in 30 years. While the owner is paying down the mortgage, the return on equity
    is the principal payment divided by the equity.  The principal payment is the
    net rent minus non-financing costs and interest, so it is actually a profit. The
    initial return on equity is 6.321 % APR, or 6.507 % APY.  This is calculated by
    dividing the $ 210.70 monthly principal payment by the initial $ 40,000 equity,
    and converting from monthly return to annual return. After 30 years, the return
    on equity is 4.864 % APR, or 4.974 % APY.  This is calculated by dividing the
    $ 810.70 monthly cash flow (which is no longer reduced by mortgage payments) by
    the $ 200,000 equity after 30 years, and converting from monthly return to annual
    return. The cap rate is the same as the return on equity in the absence of debt.  In
    this example, 4.864 % APR, or 4.974 % APY. The return on equity declines from
    6.507 % APY initially to 4.974 % APY after 30 years. This is because the cap rate
    exceeds the note rate (4.974 % APY vs. 4.594 % APY), and the leverage decreases
    from 5x to 1x. The weighted average compound annual growth rate of the equity
    during the 30 years is 5.511 % APY.  Per the original poster''s answer, this is
    computed by taking the 30th root of the 5-fold increase in equity.  Because the
    owner made no extra principal payments (besides those already discussed), the
    relevant amounts are the initial $ 40,000 owner payment and the final $ 200,000
    owner equity.  5.511 % APY corresponds to a 5.377 % APR. The internal rate of
    return if the owner never sells can be computed by treating the deal as a $ 40,000
    up-front payment, in exchange for an $ 810.70 monthly payment starting in 30 years.  The
    internal rate of return (IRR) is not a very useful number, because it assumes
    that you can somehow reinvest the eventual dividends at the same rate.  In this
    example, the IRR is 5.172 % APR, or 5.296  % APY.  In this example, the IRR is
    calculated by (iteratively) finding an interest rate for which (initial investment)
    * (1 + IRR) ^ (number periods before dividends start) = (periodic dividend) /
    (IRR - growth rate of dividend).  For example: $ 40,000 * (1.004309687)^360 =
    $ 810.70 / (0.004309687 - 0) = $ 188,111 I then converted the 0.431 % monthly
    IRR to an annual IRR. The deal can be thought of as a return on equity, plus a
    return on paying down the mortgage. When computing the return from paying down
    the mortgage, the initial equity is irrelevant.  It does not matter whether you
    start with a $ 160,000 mortgage on a $ 160,000 property, a $ 160,000 mortgage
    on a $ 200,000 property, or a $ 160,000 mortgage on a $ 1,000,000 property.  All
    that matters is the note rate on the mortgage, which is the applicable compound
    interest rate. The return on paying down the mortgage equals the note rate of
    the mortgage.  For a 4.5% note rate, this works out to a 4.594% annual percentage
    yield (APY). You can confirm this by looking at your amortization schedule.  Suppose
    you have a $ 160,000 mortgage with a fixed 4.5% APR note rate for 360 months.  Your
    monthly payment is $ 810.70.  In the first month, $ 600 goes toward interest,
    and $ 210.70 reduces the principal.  In other words, the $ 210.70 principal payment
    eliminated the need for a $ 810.70 payment 30 years later.  Notice that: . $ 210.70
    * (1 + 0.045 / 12)^360 = $ 210.70 * (1.00375)^360 = $ 210.70 * 3.8477 = $ 810.71
    which is within rounding error of $ 810.70. The interest rate is 3/8 % per month,
    which is an APR of 4.5%, and an APY of 4.594 %.'
  - It seems that you're complicating things quite a bit.  Why would you not create
    a business entity, open one or more bank accounts for it, and then have the money
    wired into those accounts?  If you plan on being a company then set up the appropriate
    structure for it. In the U.S., you can form an S-corporation or an LLC and choose
    pass-through taxation so that all you pay is income tax on what you receive from
    the business as personal income.  The business itself would not have tax liability
    in such a case. Co-mingling your personal banking with that of your business could
    create real tax headaches for you if you aren't careful, so it's not worth the
    trouble or risk.
  - No.  Mark-to-market valuation relies on using a competitive market of public traders
    to determine the share price --- from free-market trading among independent traders
    who are not also insiders.   Any professional valuation would see through the
    promotional nature of the share offer. It is pretty obvious that the purchaser
    of a share could not turn around and sell their share for $10, unless the 'free
    hosting' that is worth most of the $10 follows it... and that's more of hybrid
    of stock and bond than pure stock.  It is also pretty obvious that selling a few
    shares for $10 does not mean one could sell 10,000,000 shares for $10, because
    of the well known decreasing marginal value effect from economics. While this
    question seems hypothetical, as a practical matter offering to sell share of unregistered
    securities in a startup for $10 to the general public,  is likely to run afoul
    of state or federal securities laws -- irregardless of the honesty of the business
    or any included promotional offers.   See http://www.sec.gov/info/smallbus/qasbsec.htm
    for more information about the SEC regulations for raising capital for small businesses.
- source_sentence: Why can't the Fed lower interest rates below zero?
  sentences:
  - '"Keep in mind that the Federal Reserve Chairman needs to be very careful with
    his use of words.  Here''s what he said: It is arguable that interest rates are
    too high, that they are being constrained by the fact that interest rates can''t
    go below zero. We have an economy where demand falls far short of the capacity
    of the economy to produce. We have an economy where the amount of investment in
    durable goods spending is far less than the capacity of the economy to produce.
    That suggests that interest rates in some sense should be lower rather than higher.
    We can''t make interest rates lower, of course. (They) only can go down to zero.
    And again I would argue that a healthy economy with good returns is the best way
    to get returns to savers. So what does that mean? When he says that ""we can''t
    make interest rates lower"", that doesn''t mean that it isn''t possible. He''s
    saying that our demand for goods is lower than our ability to produce them. Negative
    interest would actually make that problem worse -- if I know that things will
    cost less in a month, I''m not going to buy anything. The Fed is incentivizing
    spending by lowering the cost of capital to zero. By continuing this policy, they
    are eventually going to bring on inflation, which will reduce the value of the
    currency -- which gives people and companies that are sitting on money an dis-incentive
    to continue hoarding it."'
  - '"I used to use etfconnect before they went paid and started concentrating on
    closed end funds. These days my source of information is spread out. The primary
    source about the instrument (ETF) itself is etfdb, backed by information from
    Morningstar and Yahoo Finance. For comparison charts Google Finance can''t be
    beat. For actual solid details about a specific ETF, would check read the prospectus
    from the managing firm itself. One other comment, never trust a site that ""tells
    you"" which securities to buy. The idea is that you need sources of solid information
    about financial instruments to make a decision, not a site that makes the decision
    for you. This is due to the fact that everyone has different strategies and goals
    for their money and a single site saying buy X sell Y will probably lead you to
    lose your money."'
  - '"It''s actually the other way around.  Distributions in an LLC are usually based
    on  each member''s equity share, although the operating agreement can specify
    how often such distributions are made.  Shareholders in a corporation can receive
    dividends, but those are determined by the corporation''s board and can vary depending
    on the class of stock each shareholder owns.  Preferred-class shareholders, who
    may hold a smaller overall fraction of the company''s outstanding shares than
    the common stock shareholders, may receive disproportionately larger dividends
    per share than common stock shareholders, which is one of the (many) reasons that
    preferred stock is a better choice when it is available.  Take, for instance,
    what Berkshire Class ""A"" shareholders receive in dividends per year compared
    to Class ""B"" shareholders. Here''s a good link from LegalZoom that can explain
    what you''re asking about: Explanation of LLC distributions I hope this helps.
    Good luck!"'
- source_sentence: Why can Robin Hood offer trading without commissions?
  sentences:
  - They make money off you by increasing the spread you buy and sell your stocks
    through them.  So for example, if the normal spread for a stock was $10.00 for
    a buy and $10.02 for a sell, they might have a spread of $9.98 for the buy and
    $10.02 for the sell. So for an order of 1000 shares (approx. $10000) they would
    make $0.02 per share which would equal $20.00.
  - As I said in the comments, from the SMH article, you will get $3.30 per share
    you hold in Wotif. The bit about Wotif veing replaced in the S&P ASX200 index
    by another company has no impact on your shares in Wotif. It just means that the
    index (the amalgamation of 200 companies) will have one drop out (Wotif) and another
    replace it (Healthscope).
  - '"To answer this part of the question: ""How can you build an index based on shipping
    routes - what is the significance of that? Indexes are traditionally built based
    on companies: e.g. S&P Index is a basket of companies whose price varies. But
    here you need a basket of FFA contracts from different oil firms (Shell, BP),
    5 year Shell FFA''s, 10 year shell FFA''s. Where do routes enter the picture?
    Let the tanker any route he feels like."" No, you don''t get a basket of FFA contracts
    from given companies (such as Shell and BP). What you get are rates assessed by
    a panel of brokers for the main tanker routes (especially in the tanker market,
    there are comparatively few standard routes, because the major oil loading areas
    are also comparatively few). The panel will assess the spot and future markets
    on a daily basis, and issue the rates accordingly."'
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
    'Why can Robin Hood offer trading without commissions?',
    'They make money off you by increasing the spread you buy and sell your stocks through them.  So for example, if the normal spread for a stock was $10.00 for a buy and $10.02 for a sell, they might have a spread of $9.98 for the buy and $10.02 for the sell. So for an order of 1000 shares (approx. $10000) they would make $0.02 per share which would equal $20.00.',
    'As I said in the comments, from the SMH article, you will get $3.30 per share you hold in Wotif. The bit about Wotif veing replaced in the S&P ASX200 index by another company has no impact on your shares in Wotif. It just means that the index (the amalgamation of 200 companies) will have one drop out (Wotif) and another replace it (Healthscope).',
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


* Size: 5,498 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 4 tokens</li><li>mean: 15.41 tokens</li><li>max: 36 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 110.7 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  |:-------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Real estate loans for repairs</code>                                                             | <code>If you intend to flip this property, you might consider either a construction loan or private money. A construction loan allows you to borrow from a bank against the value of the finished house a little at a time. As each stage of the construction/repairs are completed, the bank releases more funds to you. Interest accrues during the construction, but no payments need to be made until the construction/repairs are complete.  Private money works in a similar manner, but the full amount can be released to you at once so you can get the repairs done more quickly. The interest rate will be higher. If you are flipping, then this higher interest rate is simply a cost of doing business. Since it's a private loan, you ca structure the deal any way you want. Perhaps accruing interest until the property is sold and then paying it back as a single balloon payment on sale of the property. To find private money, contact a mortgage broker and tell them what you have in mind.  If you're intending to keep the property for yourself, private money is still an option. Once the repairs are complete, have the bank reassess the property value and refinance based on the new amount. Pay back the private loan with equity pulled from the house and all the shiny new repairs.</code> |
  | <code>Do Options take Dividend into account?</code>                                                    | <code>No can't make quick bucks. It depends very much on what the strike price was. Dividends which are below 10% of the market value of the underlying   stock, would be deemed to be ordinary dividends and no adjustment in   the Strike Price would be made for ordinary dividends. For   extra-ordinary dividends, above 10% of the market value of the   underlying security, the Strike Price would be adjusted. Refer more at NSE India Edit: The Nifty consists of 50 stocks. The largest one has weight of around 8%. So 10% on this will only translate to .8% on index.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
  | <code>If you want to trade an equity that reflects changes in VIX, what is a good proxy for it?</code> | <code>"There is no good proxy for VIX, because it is a completely made-up value. Most listed options trade on an underlying security.  I can therefore choose to buy either the stock, or a future or option on that stock.  In this way, the future and option are derivatives in that they derive their value (in part) based on something else, in this case the stock price as of now. VIX is a different entity altogether.  It is based on the volatility of the market, using ""market expectation of near term volatility conveyed by stock index option prices"".  But the FAQ goes on to state that they are adding factors into the formula. So right away there is no one equity/stock that you can hold that will necessarily match the VIX in any significant way, because it is not directly based on stocks, but indirectly through other options and computations. In effect, therefore, the VIX in indeed only available through its options, and is not observable (tradable) in and of itself."</code>                                                                                                                                                                                                                                                                                                       |
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
- `fp16`: False
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
- Python: 3.12.12
- Sentence Transformers: 3.0.1
- Transformers: 4.43.3
- PyTorch: 2.3.1
- Accelerate: 1.12.0
- Datasets: 4.4.2
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