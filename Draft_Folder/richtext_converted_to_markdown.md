MASTER PLAN: Router-Based Retention in a Sequence-Level Attention-to-SSM Hybrid
===============================================================================

**Version:** Master Plan V10 (merges V7, V8, V9, the two review documents, the explanation document, and my reading of the papers)**Format note:** This is written as plain chat text so you can paste it into your own Word file. I did not create a file.**Assumption:** Nothing has been coded yet. If you have code, map it to Sections 7 and 16.

0\. How to use this document
----------------------------

*   **Sections 1-5:** what the project is, why it exists, and what we claim.
    
*   **Sections 6-10:** exactly what to build (model, losses, data, evaluation).
    
*   **Section 11:** why the project might fail, and the fixes built into this plan.
    
*   **Sections 12-16:** tests, phases, gates, timeline, code layout.
    
*   **Sections 17-18:** how to write the paper and report results.
    
*   **Sections 19-20:** FAQ and glossary. Check these first when you have a doubt.
    

**What changed from V9 in this version:**

1.  A failure analysis (Section 11) with fixes.
    
2.  A third training configuration (S2) so the router can learn retention from the language-model loss.
    
3.  A rule for the last prefix block.
    
4.  A "calibrated probe" rule for the hindsight loss.
    
5.  A minimum publishable paper (MVP) defined in advance.
    
6.  Three literature checkpoints with pivot plans.
    
7.  The evaluation harness as its own deliverable.
    

1\. The problem in plain language
---------------------------------

Language models read long text. Two designs dominate.

Transformer (attention)Mamba (state space model, SSM)Memory of the pastKeeps every past token in the **KV cache**Squeezes the past into one fixed-size **state**Memory costGrows with text lengthConstantExact recall of a detail from far backExcellentWeak, because the state is finite

Analogy: attention keeps every page of the book on your desk. Mamba keeps one page of notes.

**The core goal of the project (from the explanation document):** keep attention-like retrieval without paying for a KV cache over the whole sequence.

**Three connected sub-problems:**

*   **A. Attention is expensive.** Dense attention over the early part of the text costs a lot. Sparse (block-selected) attention reduces that.
    
*   **B. Old memory stays around.** After switching to Mamba, the old KV cache should be deleted ("purged") so memory stops growing.
    
*   **C. Can the compact state remember what matters?** After purging, you can't go back. If the state forgot something, it is lost.
    

**The scientific question underneath (added during review):** when we delete most of the old memory, **which pieces should we keep, and who should decide?** In particular, can the same router that chooses where to attend also choose what to keep?

2\. Background you need
-----------------------

### 2.1 TransMamba (the starting point)

*   The sequence is split at a **TransPoint (P)**. Before P, attention. After P, Mamba.
    
*   A **Memory Converter** turns the attention side's K and V information into Mamba's starting state.
    
*   Each layer can have its own P.
    

### 2.2 Sparse attention (what we borrow)

*   Split the early text into **blocks**. Score blocks. Attend only to the top few.
    
*   Related methods: NSA and MoBA (trainable block-sparse attention).
    

### 2.3 KV-cache eviction (what we borrow)

*   Deciding which old tokens to keep. Methods: H2O (keep tokens that received lots of attention), SnapKV, learned indexers (IndexMem), and others.
    

### 2.4 Key terms used throughout

*   **Prefix:** tokens before P. **Suffix:** tokens after P.
    
*   **Block:** 32 consecutive tokens.
    
*   **Router:** a small network that scores blocks for a query.
    
*   **Retained blocks (m):** how many prefix blocks we keep after the switch.
    
*   **Purge:** deleting everything in the old cache except what we retain.
    
*   **Hindsight:** using the future (during training only) to label which blocks were actually needed.
    
*   **Oracle:** the best possible choice, computed using the future. It is only a ceiling for comparison.
    

3\. History of the plan and what each stage taught us
-----------------------------------------------------

StageWhat it wasWhat went wrong or what we learned**V7**Sparse-TransMamba: block-sparse prefix attention, converter state, residual scaling alpha, KV-Purge, load-balancing loss, many correctness testsPurge was meaningless (the suffix never looked at the old KV, so purge changes nothing). Claim of dense gradients to the router was wrong. Convert() had no decay. Balancing loss pushes against peaked retrieval. Floor and fallback thresholds contradicted each other. T4 hardware claims were off**V8**Budgeted-TransMamba: router doubles as eviction policy, keep m blocks, random budget training, capacity studyReviews showed the pieces are already known. The router's retention usefulness was assumed, not tested**V9**Hindsight-trained router, three-arm comparison, local window, synthetic regimes, one backbone with many retention policiesAfter reading the papers: HAM, IndexMem, Oryx overlap strongly. Novelty must be an empirical study, not a new mechanism**V10 (this)**V9 plus failure analysis and fixes, S2 configuration, MVP definition, checkpoints

### 3.1 Traceability: every V7 item and where it went

V7 itemStatus in V10Static per-layer TransPoint, terminology guardKept. Main runs use one shared P. Per-layer schedules are an ablationMoE-style soft top-k block selectionKept, implemented as a logit bias log(w) so the router gets gradientState-level residual scaling alphaMoved to an ablation (legacy converter)Converter state h\_PReplaced by the true SSM scan state at P. Legacy converter kept as an ablationLog-space decay injectionNot needed in the main model (the SSM runs over the whole sequence). Used only in the legacy ablationKV-PurgeKept and made meaningful: purge keeps m blocks, a window, and sink tokensLoad-balancing/entropy loss with 3-value sweepOff by default. Used only if router collapse appears, with the sweep {1e-3, 5e-3, 1e-2}T ≤ P edge caseKept (test 3)Train/inference equivalence testKept (test 2)Checkpointing equivalence testKept (test 8)Gradient flow testKept (test 7)Fallback threshold measurementKept in Phase 0, reported honestly even if sparse is never fasterSchedule ablations (decreasing, increasing, U-shaped, shared)Kept in Phase 5Baselines (TransMamba, Transformer, Mamba, interleaved)Kept, plus sliding-window+SSMDiagnostics (entropy, gradient norms, NaNs, sparse fraction)KeptDual T4, DDP, fp16Kept (bf16 and pooled 30 GB claims removed)Seeds (3 for main, 1 for grid)KeptNon-goalsKept and extended (Section 5.4)Required paper sentencesKept and updated (Section 17.4)

4\. Related work and the honest novelty position
------------------------------------------------

### 4.1 What I read and how closely

PaperHow I read itWhat it doesRelevance to us**TransMamba**Described in your documents and cited textAttention-to-Mamba switch at a fixed TransPoint with a Memory ConverterOur starting point. Not our novelty**Oryx** (arXiv 2605.28769)Full textSwitches mixers along the sequence. Ties K and V across mixers, separate queries. Keeps BOTH KV cache and state. Says reducing that overhead is openSequence-axis switching is not new. Bounded memory after the switch is the open gap we target. Gives design rules**HAM** (Zyphra, 2603.22325)Abstract and project page onlyRNN compresses all; the KV cache keeps only tokens the RNN can't predict (learned router)Closest to "state-aware retention". Run it as a baseline arm, not a novelty**IndexMem** (arXiv 2605.25475)Full method and experiment sectionsLearned indexer distilled from backbone attention (max over queries), plus a gated latent memory for evicted tokensOur "separate scorer" arm. Also shows a hindsight-style target is not new**LORE-KV** (2610.07643)Abstract onlyTraining-free future-aware eviction using sampled continuations and leave-one-out deletion costSupports the premise that "attention seen" is not "future use". We adopt leave-one-block-out loss as the oracle**MaskAhead** (2610.06996)Abstract onlyOne ranking mechanism for selection and eviction, in block diffusion models, training-freePrecedent for unified selection/eviction elsewhere**laLTE** (2510.20787, ICLR 2026)Abstract and intro onlyLinear attention + sparse attention + learned token evictionLearned eviction in a hybrid is established**ARM** (2609.24417)Confirmed it exists, did not read itDescribed in the review as routed memory with variable memory useMust read before writing related workNSA, MoBA, H2O, SnapKVGeneral knowledgeBlock-sparse attention; heuristic evictionStandard backgroundAttention-Gate, KVP, "Make Each Token Count", DeltaSOnly the reviewers' descriptions, not verifiedLearned eviction variants; DeltaS is streaming videoVerify before citing

### 4.2 What the two review documents said, and my verdict

*   **Review 1 (harsh):** the plan is not novel as written. Core ingredients exist. Shared-router idea is the only part with room. Needs a three-scorer test and a test of whether attention relevance predicts future use. **My verdict:** mostly correct. I adopted the three-scorer comparison, the relevance-versus-future-utility measurement, and the hindsight objective. Its numeric ratings (5/10, 7/10) are opinions, not measurements.
    
*   **Review 2 (prior-art search):** no exact duplicate found, but crowded territory (Oryx, laLTE, IndexMem, MaskAhead, LORE-KV, DeltaS). **My verdict:** consistent with what I verified. "I found no exact match" is weak evidence, because search coverage is limited.
    
*   **Explanation document:** gave the clean problem statement and the best-case and negative-case outcomes, which are used in Section 14.
    

### 4.3 Honest novelty statement

All mechanism pieces are known: sparse routing, learned eviction, hybrid switching, state-informed retention, and future-aware targets. What I did **not** find in what I read is a **controlled, from-scratch study in an autoregressive attention-to-SSM hybrid** that asks when one router can serve both selection and retention, using designed demand regimes and oracle ceilings. That is the defensible contribution. A reviewer may still call it incremental. The defense is rigorous controlled comparisons and a reusable evaluation harness.

5\. Aim, research question, hypotheses, claims
----------------------------------------------

### 5.1 Aim

Study how a hybrid model should use bounded memory after switching from attention to a recurrent state, and whether one router can decide both where to attend and what to keep.

### 5.2 Research question

**After a hybrid switches from attention to a recurrent state, can the same learned router that decides where to attend also decide what to retain, and when does sharing succeed or fail?**

### 5.3 Hypotheses (written before running; results confirm or refute them)

*   **H1:** Keeping m>0 prefix blocks beats pure purge (m=0) on recall, including against a sliding-window baseline with the same total memory.
    
*   **H2:** A router trained only for attention selection beats random and recency for retention when future demand is visible in the prefix, and roughly ties H2O-style scoring there.
    
*   **H3:** When future demand is **not** visible to prefix attention (regime R4), the attention-trained router fails and a hindsight-trained scorer succeeds.
    
*   **H4:** A shared router trained with the hindsight objective matches a separate scorer trained on the same objective, with fewer parameters and small perplexity cost.
    
*   **H5:** Recall capacity grows predictably with the number of retained blocks and the state size.
    

### 5.4 Contributions (in order of importance)

1.  A controlled comparison of retention scorers (shared vs separate vs heuristic vs oracle) on one fixed backbone.
    
2.  A characterization of when attention routing transfers to retention, using four designed demand regimes.
    
3.  A reusable evaluation harness (RetentionBench): regimes, oracles, policy interface, metrics.
    
4.  Recall-vs-memory curves with memory-matched baselines.
    
5.  An empirical capacity law for state plus retained blocks.
    

### 5.5 Allowed and forbidden claims

**Allowed (if supported):** items 1-5 above.**Forbidden:**

*   New hybrid architecture, sequence-axis switching, sparse attention, learned eviction, variable budgets, or initial-state injection as novelties.
    
*   "Constant memory" before or after the switch. Say **bounded and tunable**.
    
*   Any wall-clock speedup unless measured and positive.
    
*   Any claim that the original Memory Converter loses information (kept from V7 non-goals).
    
*   No claim that FLOPs alone yield an optimal P (kept).
    

**Non-goals (V7 kept, plus new):** no dual normalization, no adaptive temperature, no mixture-of-experts, no learned or input-dependent TransPoint, no full multi-branch NSA, no output-level residual scaling default, no custom CUDA kernels.

6\. System overview (the idea in one picture, in words)
-------------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Tokens 0..P-1 (prefix)                 Tokens P..T-1 (suffix)  Router-selected sparse attention       SSM (Mamba-like) + small window attention          |                                   + attention to m retained prefix blocks          v                                          ^  SSM also scans the prefix  ---- state S_P ---------+          |  At position P: choose m blocks to keep, purge the rest   `

After the switch, each layer keeps three memories:

1.  the **SSM state** (fixed size),
    
2.  a **window** of the last 128 tokens plus 4 sink tokens,
    
3.  **m retained blocks** chosen from the prefix.
    

**Experiment in one sentence:** train one backbone, then test many different rules for choosing the m blocks, including our shared router, and compare them to ceilings and baselines.

7\. Detailed model specification
--------------------------------

### 7.1 Configuration (main model)

ItemValueParametersabout 10M including embeddings. Report the non-embedding count tooTokenizerBPE, 14,336 text tokens plus 2,048 reserved ids (total 16,384). Reserved: 1,024 KEY, 1,024 VALUE, plus PAD, EOS, SEP, QUERY, FLAGLayers L8d\_model256Heads H / head dim4 / 64FFNSwiGLU, hidden 704. RMSNorm pre-norm. Tied embeddingsTrain length T2048TransPoint P1024 (same for all layers in main runs)Block size B32, so the prefix has 32 blocksPrefix selectionk\_sel = 8 previous blocks per query, plus own block (causal) and 4 sink tokensSuffix window W128 tokens, plus 4 sink tokensSSMMamba-2-style scalar decay per head, d\_k = d\_v = 64, chunk size 64Router2 router heads, d\_r = 32, per layerPrecisionfp16 autocast with GradScaler. SSM decay and state in fp32. No bf16 on T4

### 7.2 Memory accounting (use everywhere)

*   State per layer = H x d\_k x d\_v = 16,384 floats.
    
*   One cached token (K and V) per layer = 2 x H x d\_h = 512 floats.
    
*   So **one state equals 32 token-equivalents, exactly one block**.
    
*   Memory after P per layer, in token-equivalents = 32 (state) + W + 4 + 32 x m. The conv buffer (3 tokens) is negligible.
    
*   All baselines are matched on this number.
    

### 7.3 Layer forward pass (training mode), step by step

Input x: \[batch, T, D\].

**Step 1. Projections.**

*   h = RMSNorm(x).
    
*   k = W\_k h and v = W\_v h (tied across the attention and SSM sides, Oryx rule). Apply short causal depthwise convolution (kernel 4) plus SiLU to k and v.
    
*   q\_a = W\_qa h (attention query), q\_s = W\_qs h (SSM query). Separate queries, because tying hurt in Oryx.
    
*   dt = W\_dt h, shape \[batch, T, H\]. Gate g = SiLU(W\_g h).
    
*   Split into heads. Apply RoPE to q\_a, q\_s and k.
    

**Step 2. Edge case.** If T ≤ P, use prefix-mode attention for the whole sequence and ignore the SSM output. No NaNs allowed (test 3).

**Step 3. Router (prefix queries).**

*   Query side: u\_i = router\_q(h\_i) of shape \[H\_r, d\_r\]. Head gate gamma\_i = sigmoid(router\_g(h\_i)).
    
*   Block side: z\_b = router\_k(RMS-normalized mean of the block's keys), shape \[d\_r\].
    
*   Score r(i, b) = sum over router heads of gamma\_i,h times dot(u\_i,h, z\_b,h) divided by sqrt(d\_r).
    
*   Eligibility: block b is eligible for query i only if b < block(i). The query's own block is always attended causally. In block 0 there is nothing to select.
    

**Step 4. Prefix sparse attention.**

*   For each query take the top k\_sel eligible blocks. Weights w = softmax of r over the selected blocks.
    
*   Allowed keys = own block (causal) OR sink tokens OR selected blocks.
    
*   Logit bias log(w) on selected-block tokens. Own block and sink get bias 0.
    
*   Run torch scaled\_dot\_product\_attention with a float additive mask. Every query row must have at least one allowed key.
    
*   Implemented as **masked dense attention** (no custom kernel). FLOPs are reported analytically.
    
*   **Gradient fact:** unselected blocks get no gradient from the LM loss. The router learns through selected blocks (via log w) and through the hindsight loss.
    

**Step 5. SSM over the whole sequence.**

*   Decay a\_t = exp(-softplus(dt\_t + dt\_bias\_h) x exp(A\_log\_h)), in fp32.
    
*   Recurrence S\_t = a\_t S\_{t-1} + k\_t^T v\_t. Output y\_t = q\_s,t S\_t.
    
*   Chunked algorithm (chunk 64): intra-chunk quadratic term, chunk-end states, state passing, inter-chunk term. All decay products come from a cumulative sum of log a in fp32.
    
*   P = 1024 is a chunk boundary, so S\_P is the chunk-boundary state. Prefix outputs of the SSM are discarded.
    
*   Because the SSM scans the prefix, the state at P is exact. No initial-state injection is needed.
    

**Step 6. Retained set (depends on prefix tokens only).**

*   **Candidates:** prefix blocks that have at least one later prefix query, which are blocks 0 to 30. The final block 31 is covered by the window for the first W suffix tokens and is not a retention candidate. (Fix F3.)
    
*   Retention logit R\_b = logsumexp over prefix queries i with block(i) > b of r(i, b). Variants evaluated at inference: max, and mean selection weight.
    
*   Retained set = top-m candidates by R\_b.
    
*   Training samples m uniformly from {0, 1, 2, 4, 8, 16, 31} per batch (same across layers).
    
*   Evaluation uses the same grid. Also run a **full-KV reference mode**: the suffix attends to window plus all prefix tokens (the true no-purge upper bound).
    

**Step 7. Suffix attention.**

*   Suffix query at position t may attend to: tokens in (t - W, t\], OR the 4 sink tokens, OR tokens in retained blocks. Remove duplicates between the window and retained blocks.
    
*   y\_suffix = y\_ssm + beta\_h x y\_attn\_suffix, with a learned per-head scalar beta\_h initialized to 1.
    

**Step 8. Output.** z = concat(y\_prefix\_attention, y\_suffix). Apply GatedRMSNorm with gate g, output projection W\_o, residual, then the FFN with its own residual.

### 7.4 Layer variants (one code path, config switch)

NameDescriptionRoletm\_fullEverything aboveOur modelswa\_mPrefix uses sliding-window attention instead of the router; suffix has the window plus SSM; no retention. Trained with random window size per batch in {128, 160, 192, 256, 384, 640, 1024}Memory-matched baseline (Samba-style)dense\_attnOrdinary causal dense attention in every layerTransformer referencessm\_onlySSM everywhereMamba referenceinterleavedAlternate dense\_attn and ssm\_only layersV7 interleaved hybrid baseline. Memory not bounded, quality reference onlytm\_legacyPrefix dense attention, closed-form converter from K,V with alpha (init 0.5), suffix SSM from converted state, no window or retentionTransMamba-style re-implementation (label it as such, not official code)

### 7.5 Inference procedure and cache

Per layer cache: KV buffer (while position ≤ P), SSM state, conv buffer (last 3 tokens), rolling window buffer, retained blocks with absolute positions, sink tokens.

*   **Before P:** standard KV-cache decoding with the same sparse selection. Update the SSM state and conv buffer each token.
    
*   **At P:** compute R\_b from prefill router scores. Keep the top-m blocks. Purge all other prefix KV except the window buffer and sink tokens.
    
*   **After P:** one recurrent SSM step per token, attention to window plus retained blocks plus sink.
    
*   RoPE uses stored absolute positions.
    
*   **Equivalence requirement:** for any fixed m, teacher-forced incremental decode equals the full training forward within fp32 tolerance 1e-4 and fp16 tolerance 1e-2. This holds because the retained set uses prefix information only.
    
*   Memory after P is **bounded and tunable**, not constant.
    

### 7.6 Notes on things we deliberately simplified

*   No per-layer P in main runs (ablation only).
    
*   No custom sparse kernel. The honest statement is that masked dense attention on T4 at this scale will not be faster than dense, so we report analytic FLOPs and memory, not speed.
    
*   The block score uses mean-pooled keys. This is simple and known to be weak for fine retrieval; the hindsight objective is partly meant to compensate.
    

8\. Training objectives and run configurations
----------------------------------------------

### 8.1 Language-model loss

Standard next-token cross-entropy over all positions. For synthetic data, value positions are upweighted by 5.

### 8.2 Hindsight target (no gradient)

*   Sample 64 random suffix positions per sequence ("probe queries").
    
*   For each probe query j, compute dense attention probabilities of detached q\_a,j over all keys at positions ≤ j (prefix plus earlier suffix).
    
*   Sum the mass falling on each prefix block, **excluding the 4 sink tokens** (Fix F4), average over probes and heads. Result y\_b over blocks.
    
*   Normalize y over candidate blocks to sum to 1.
    

### 8.3 Hindsight loss

*   KL(y || softmax(R)) over candidate blocks, summed over layers, averaged over the batch.
    
*   Gradient flows only into router parameters (inputs detached), so it cannot distort the backbone representation. Ablation: allow gradient into hidden states.
    
*   **Calibrated-probe rule (Fix F1):** apply the hindsight loss only on steps where the sampled m ≥ 8 (about 3 of 7 values). Reason: with small m, suffix queries rarely see most blocks, so their dense attention on unseen blocks is unreliable. Ablation: apply always.
    
*   Total loss = LM loss + lambda\_h x hindsight loss. Sweep lambda\_h in {0, 0.03, 0.1, 0.3}.
    

### 8.4 Separate scorer head (always attached)

*   Same architecture and size as the router but with its own parameters, inputs detached, trained only on the hindsight loss with weight 1. It never affects the backbone.
    
*   This is the IndexMem-style learned scorer arm.
    

### 8.5 Training configurations that matter

NameRouter trained byQuestion it answers**S0**LM loss only (selection-only), lambda\_h = 0Does an attention-trained router already transfer to retention?**S1**LM loss plus hindsight loss (lambda\_h chosen from the sweep)Does the explicit demand objective help, and what does it cost?**S2** (Fix F2)LM loss, with the retained blocks' attention logits receiving a bias alpha x (R\_b - max R), alpha learnable, init 1; no hindsight lossCan the router learn retention end-to-end from the LM loss alone?

S2 is run as one-seed ablation first. If S1 fails but S2 works, S2 is promoted to a main arm.

### 8.6 Other training rules

*   Optimizer AdamW (0.9, 0.95), weight decay 0.1, warmup 300 steps, cosine decay to 10%, gradient clip 1.0.
    
*   32k tokens per step (8 sequences x 2048 tokens per GPU, 2 GPUs). About 3,000 steps for 100M tokens.
    
*   Learning rate chosen per architecture from a short sweep {5e-4, 1e-3, 2e-3} on 20M tokens.
    
*   Auxiliary balancing loss only if router collapse is observed (random-block warmup for the first 5% of steps first).
    
*   Checkpoint every 500 steps. Runs must be resumable.
    

9\. Data
--------

### 9.1 Text

*   FineWeb-Edu sample (sample-10BT). Tokenize about 150M tokens with the 16k BPE into uint16 shards, concatenated with EOS, chunked to 2048. Hold out 5M tokens for validation.
    
*   Train the tokenizer on about 200 MB of the same text.
    
*   Save shards as a Kaggle dataset so you never re-tokenize.
    

### 9.2 Synthetic recall data (15% of the pretraining mix, plus separate evaluation sets)

Format (length 2048):

*   Prefix (positions < 1024) holds N key-value pairs as \[KEY, VALUE\], placed at random positions among filler tokens.
    
*   Suffix holds Q queries as \[QUERY, KEY\]. The target is the VALUE token. N is from {8, 16, 32, 64, 128, 256}. Q = min(N, 32).
    

**Four demand regimes (how queries choose pairs):**

RegimeRulePurposePredicted behavior**R1 Uniform**Uniform over pairsControl: no scorer beats coverageAll scorers equal within noise**R2 Flagged**A FLAG token precedes some pairs; 80% of queries target flagged pairsCue visible in the prefix contentLearned scorers do well. S0 may do well if FLAG attracts attention**R3 Repetition**Some keys are mentioned several times in the prefix; queries favor themAttention mass reveals demandH2O and S0 do well**R4 Hidden**Pairs whose VALUE equals a special TOP token are queried 80% of the time; no prefix query refers back to themCurrent attention cannot see future demandH2O and S0 fail. Separate head and S1 succeed

R4 is the decisive regime. If R4 does not split the arms, debug the hindsight target before anything else.

### 9.3 Additional evaluation sets

*   **Position sweep:** pairs placed only in early, middle, or late prefix blocks (tests recency bias).
    
*   **Suffix-induction perplexity on real text:** perplexity only on suffix tokens whose preceding bigram appeared in the prefix with the same next token (a real-text recall proxy, no extra data needed).
    
*   **Length extrapolation:** T = 4096 with P = 1024 (suffix 3072). Only if time allows.
    

10\. Retention policies, oracles, metrics, harness
--------------------------------------------------

### 10.1 API requirement

The model forward must accept an optional retained\_idx of shape \[batch, layers, m\]. When given, it replaces the router's own choice. Every policy then runs on the same trained backbone, so comparisons are cheap and paired.

### 10.2 Policies (each chooses m candidate blocks per layer)

1.  Random.
    
2.  Recent: the last m candidate blocks.
    
3.  First: the first m blocks.
    
4.  H2O-style: accumulated attention mass each block received from all prefix queries (dense probe pass at purge time).
    
5.  SnapKV-style: mass from the last 64 prefix queries.
    
6.  Surprise (HAM-style proxy, training-free): per-token prediction error of the SSM state, e\_t = norm of (v\_t - k\_t S\_{t-1}), averaged per block. Label it a proxy, not HAM's learned router.
    
7.  Shared router with aggregator variants (logsumexp, max, mean weight).
    
8.  Separate scorer head (IndexMem-style).
    
9.  Oracle-Attn: blocks ranked by true suffix attention mass (uses the future, ceiling).
    
10.  Oracle-LOBO: blocks ranked by leave-one-block-out utility.
    

### 10.3 Leave-one-block-out utility

On a small evaluation subset (200 sequences per regime): with all candidate blocks retained, compute the suffix loss. Remove block b, recompute. Utility\_b = loss increase. This is a marginal (greedy) approximation; state the limitation in the paper.

### 10.4 Metrics

*   Recall accuracy per cell (regime, N, m, policy), 2,000 sequences per cell.
    
*   **Oracle gap closed:** G = (acc\_policy - acc\_random) / (acc\_oracle - acc\_random), averaged over m and N, with paired bootstrap confidence intervals over sequences.
    
*   Area under the accuracy-vs-memory curve.
    
*   Validation perplexity and suffix-induction perplexity.
    
*   Memory in token-equivalents, peak VRAM, tokens/sec (reported honestly).
    
*   **Target validity:** Spearman correlation between hindsight target y and LOBO utility, per layer. This is also the direct measurement of "current relevance versus future utility".
    
*   Router diagnostics: selection entropy, mass on top-k, gradient norms, sparse-layer fraction, NaN checks.
    

11\. Why the project might not work, and the fixes built into this plan
-----------------------------------------------------------------------

#Failure riskFix in this planHow we detect it early1**Someone already published the same study**Three literature checkpoints (before Phase 2, before Phase 4, before writing). Pivot plans in Section 14.3. Read ARM, HAM, laLTE in fullCheckpoint searches2**The hindsight target is a poor proxy** (suffix queries never trained on non-retained blocks)Calibrated-probe rule (loss only when m ≥ 8). m = 31 appears in training. Validate target against LOBO. Alternative LOBO-based target in an ablationSpearman correlation in Phase 3 and 43**Shared router cannot do both jobs**Not a bug: it is hypothesis H3/H4. We measure it with S0, S1, S2 and report negatives. A negative result is a valid outcomeR4 results in Phase 34**Router gets no LM gradient for retention**S2 configuration gives it oneS2 ablation5**10M models do not learn retrieval**15% synthetic mix. Phase 3 trains small models on synthetic only first. If weak in Phase 4, raise synthetic share to 30% or train a 25M modelPhase 3 gates and Phase 4 recall6**State alone already solves the tasks, so retention does not matter**Phase 3a picks N and d\_k so the state failsGate 17**Differences are inside noise**Noise floor measured in Phase 0. 3 seeds, paired bootstrap, thresholds set as multiples of the noisePhase 0 and statistics rules8**Sliding-window baseline wins**The paper's claim is the comparison of scorers on one backbone, not beating the window baseline. Report honestlyPhase 4 memory-matched comparison9**Engineering leaks (future information)**Causality and retention-prefix-only tests (tests 4 and 5)Phase 2 tests10**T4 compute, fp16 instability, session limits**Pure PyTorch scan, fp32 SSM state, resumable runs, MVP defined (Section 14.1)Phase 011**Last prefix block has no later queries to score it**Candidate rule: blocks 0-30 only, block 31 covered by the windowTest 612**Time lost or unclear stopping point**MVP paper and gates with explicit stop rulesSection 13

12\. Tests (all must pass before large runs)
--------------------------------------------

1.  **SSM correctness:** chunked scan equals the naive sequential recurrence (fp32, random inputs, tolerance 1e-5).
    
2.  **Train/inference equivalence:** teacher-forced incremental decode equals full forward for several m values, across the boundary P.
    
3.  **Edge case T ≤ P:** prefix-mode attention only, no NaN.
    
4.  **Causality:** changing token t must not change outputs at positions < t, including router selection. This catches leakage from mean-pooled block keys.
    
5.  **Retention uses prefix only:** changing suffix tokens must not change the retained set.
    
6.  **Budget tests:** m = 0 equals window plus state only. Full-KV reference equals no purge. Candidate rule (blocks 0-30) holds.
    
7.  **Gradient flow:** nonzero gradients reach the router (from LM loss through log w and from the hindsight loss), the SSM parameters, and the tied K/V. With inputs detached, the hindsight loss must not reach the backbone.
    
8.  **Gradient checkpointing on vs off:** same loss and gradients within tolerance. All random choices (m, probe queries) go through a seeded generator inside the checkpointed block.
    
9.  **fp16 safety:** no NaN or inf over 200 steps with GradScaler.
    
10.  **Mask sanity:** every query row has at least one allowed key. Selected-block counts equal k\_sel when enough blocks are eligible.
    
11.  **Hindsight target sanity:** target sums to 1, excludes sink tokens, identical across runs with the same seed.
    
12.  **Legacy converter:** alpha receives gradient in tm\_legacy.
    
13.  **Generator tests:** synthetic labels are correct, no train/eval seed overlap, regime statistics hold (for example 80% of R2 queries target flagged pairs).
    
14.  **Policy interface test:** passing retained\_idx equal to the router's own choice reproduces the default forward exactly.
    

13\. Phased roadmap with gates
------------------------------

**Thresholds** below are defined from the noise floor in Phase 0: a difference counts only if it exceeds three standard deviations across 3 seeds.

### Phase 0. Feasibility and noise floor (about 1 weekend)

*   Confirm two T4 GPUs, torchrun with NCCL, fp16 autocast plus GradScaler.
    
*   Measure tokens/sec for dense\_attn, ssm\_only, tm\_full at T = 2048 and rescale all budgets.
    
*   Measure sparse masked attention versus dense SDPA at 16, 32, 64 blocks (V7 fallback-threshold measurement). Report it even if sparse is never faster.
    
*   Check whether Mamba-2 or flash-linear-attention kernels run on T4 (optional; pure PyTorch scan is the default).
    
*   Train the small Transformer on 20M tokens with 3 seeds. Record the standard deviation of validation perplexity and MQAR accuracy.
    
*   **Gate:** if tm\_full tokens/sec is below 25% of the dense Transformer's, shrink the plan (fewer tokens, fewer baselines) before proceeding.
    
*   **Literature checkpoint 1** (see Section 14.3).
    

### Phase 1. Data, tokenizer, baseline infrastructure

*   Build the tokenizer, shards, synthetic generators (R1-R4, position sweep, induction mask). Pass test 13.
    
*   Implement dense\_attn and ssm\_only. Run the learning-rate sweep.
    
*   **Gate:** baselines reach sensible perplexity, and the Transformer solves MQAR for N ≤ 32 within the mixed pretraining. If not, fix data and learning rates first.
    

### Phase 2. Core implementation and tests

Build in this order: SSM chunked scan plus recurrent step (test 1), RoPE and short conv, router, block mask and sparse attention, suffix attention with window and retained blocks, cache and decode path, hindsight target and loss, override API, tm\_legacy and swa\_m.

*   **Gate:** tests 1-14 green. No large run before this.
    

### Phase 3. Synthetic capacity and regime study (the go/no-go)

Small models (2 layers, d\_model 128, T = 1024, P = 512, block 32), trained only on synthetic data.

*   **3a. Capacity:** sweep d\_k in {16, 32, 64}, m in {0, 1, 2, 4, 8}, N in {8 ... 256}, regime R1. Capacity C is the largest N with at least 90% accuracy. Expected nominal gain: each retained block holds about B/2 = 16 pairs, so retention adds about 16m pairs if the right blocks are kept. Fit and report where the law breaks.
    
*   **3b. Regimes:** train S0, S1, S2 (one seed first), and the separate head in R1-R4. Evaluate all policies.
    
*   **Gate 1:** m ≥ 4 beats m = 0 at an N where the state alone fails. If not, debug suffix retained attention.
    
*   **Gate 2:** in R4, the separate head beats H2O and random by a clear margin. If not, the target or model is broken.
    
*   **Gate 3:** in R3, H2O and learned scorers all beat random. If not, the generator is wrong.
    
*   **Literature checkpoint 2.**
    
*   **Decision:**
    
    *   R4 splits the arms and S1 matches the separate head: go to full plan.
        
    *   S0 already equals S1 everywhere: continue as "attention routing already transfers".
        
    *   Nothing beats recency/random: stop and redesign.
        

### Phase 4. Main training and evaluation (about 10M parameters, about 100M tokens per run; rescale after Phase 0)

Runs:

*   tm\_full S0 and S1 (3 seeds each), swa\_m (3 seeds), dense\_attn, ssm\_only, interleaved, tm\_legacy (1 seed each). The separate head is attached to every tm\_full run.Evaluate all of Section 10 on the S0 and S1 backbones.
    
*   Memory-matched comparison: tm\_full at budget m versus swa\_m with window W + 32m (matched token-equivalents), with ssm\_only and dense\_attn as references.
    
*   **Gate:** if tm\_full never beats memory-matched swa\_m on R2-R4, report honestly. The paper then centers on scorer comparison, capacity law and regime analysis.
    

### Phase 5. Ablations (1 seed unless a result is surprising, then 3 seeds)

*   lambda\_h in {0, 0.03, 0.1, 0.3}.
    
*   Hindsight gradient into hidden states (yes or no).
    
*   Calibrated-probe rule versus always-on.
    
*   Retention aggregator (logsumexp, max, mean weight).
    
*   Fixed m versus random m during training.
    
*   Suffix window W in {0, 64, 128, 256}.
    
*   S2 (retained-block logit bias).
    
*   Prefix attention: router-sparse, dense, sliding-window.
    
*   P schedule: shared P versus decreasing, increasing, U-shaped (per-layer P, floor above the measured fallback threshold).
    
*   Legacy converter with and without alpha versus scan-state handover.
    
*   Mixed-mode prefix (Oryx-style: with probability 0.25 replace attention with SSM outputs in random prefix chunks).
    
*   Optional second model size (25M) if Phase 4 signals are weak.
    
*   **Literature checkpoint 3** before writing.
    

### Phase 6. Analysis and paper (Section 17)

14\. Minimum paper, outcome tree, pivots
----------------------------------------

### 14.1 Minimum viable paper (MVP)

If Phase 4 fails or time runs out, the following alone is a defensible paper (workshop level):

*   RetentionBench: regimes, oracles, policy interface, metrics.
    
*   Phase 3 results: scorer comparison on small models across R1-R4, capacity law.
    
*   One set of Phase 4 runs (S0, S1, swa\_m) with fewer seeds, clearly labeled.
    

### 14.2 Outcome tree

ResultPaper framingS1 shared matches separate head, both beat H2O in R4, small perplexity cost"One router can serve selection and retention when trained with a demand objective" (strongest)S0 already matches S1 in all regimes"Attention routing already predicts future demand in these settings" (narrower empirical study)S0 fails R4, S1 fixes it but costs perplexity"Selection and retention conflict; a small hindsight objective resolves it at a measurable cost"Shared loses to separate head even with the objective"Sharing fails; retention needs its own scorer" (negative result with regime analysis)Nothing beats recency or randomStop and redesign

### 14.3 Literature checkpoints and pivots

At each checkpoint, search for new work on: learned KV retention in hybrid models, shared selection and eviction routers, bounded-memory attention-to-SSM switching. Read ARM, HAM, laLTE in full at checkpoint 1.

*   If a paper does the same shared-vs-separate controlled study: pivot to **Pivot A** (capacity-law and regime-benchmark paper, emphasizing RetentionBench) or **Pivot B** (the negative or conflict analysis).
    
*   If a paper only covers part of it: cite it, narrow the claim, and continue.
    

15\. Compute, logistics, timeline
---------------------------------

*   Kaggle free GPU quota and session limits change. Verify them. Plan for resumable runs of a few hours.
    
*   Counting the main runs: 3 (S0) + 3 (S1) + 3 (swa\_m) + 4 single baselines = 13, plus ablations. If a 100M-token run takes more than about 2 hours, reduce to 60M tokens or cut ablations. Phase 0 tokens/sec decides.
    
*   Phase 3 runs are tiny. Do them first and repeat freely.
    
*   After each run, save checkpoints and JSONL logs as a Kaggle dataset version.
    

**Suggested timeline (adjust to your deadline):**

WeeksWork1Literature refresh and Phase 02-4Phase 1 and Phase 25-6Phase 3 (go/no-go)7-9Phase 410-11Phase 512-14Analysis and writing

16\. Repository layout and coding rules
---------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/        tokenizer.py, prep_fineweb.py, synthetic.py, dataset.py  model/       rope.py, conv.py, ssm.py, router.py, attention.py, layer.py, model.py, cache.py  train/       train.py, hindsight.py  eval/        policies.py, oracle.py, recall.py, perplexity.py  tests/       one file per test in Section 12  configs/     one YAML per run (model, data, train, seeds)   `

**Rules:**

*   Every random choice (m, probe queries, regime sampling) goes through an explicitly passed, seeded generator.
    
*   SSM math in fp32; cast outputs back to fp16.
    
*   Log every evaluation cell as one JSONL line with config hash and seed.
    
*   Pre-register: before any Phase 3 or 4 run, write hypotheses, predicted outcomes, and gate thresholds into a file in the repo and commit it.
    

17\. The paper
--------------

### 17.1 Structure

1.  Introduction (problem, question, contributions).
    
2.  Related work (matrix from Section 4).
    
3.  Method (Section 7).
    
4.  Training objectives and arms (Section 8).
    
5.  RetentionBench (regimes, policies, oracles, metrics).
    
6.  Experiments: capacity, regimes, main comparison, ablations.
    
7.  Analysis: when sharing works and fails, target validity.
    
8.  Limitations and conclusion.
    

### 17.2 Figures and tables

*   Recall versus memory (token-equivalents) per regime, all policies, swa\_m, oracles.
    
*   Oracle-gap-closed bars per regime with confidence intervals.
    
*   Capacity heatmap (N at 90% versus m and d\_k) with the fitted law.
    
*   Target validity by layer (hindsight target versus LOBO).
    
*   Perplexity cost of the hindsight loss.
    
*   Memory table in token-equivalents; tokens/sec and VRAM table (honest).
    
*   Prior-art matrix and the design-space table (switch type, attention type, handover type, retention type).
    

### 17.3 Reporting rules

*   3 seeds for the main comparison, mean and standard deviation. 1 seed elsewhere, labeled.
    
*   Paired bootstrap over sequences for comparisons on the same backbone.
    
*   Say "bounded", not "constant". No speed claims unless measured.
    

### 17.4 Required sentences

*   "The TransPoint P is a static, per-layer hyperparameter fixed before training; it is neither learned nor input-dependent."
    
*   "We study whether a single learned router can serve both sparse attention selection and post-switch retention, and when sharing fails."
    
*   "Memory after the switch is bounded and tunable via the number of retained blocks."
    
*   "Sparse attention is implemented with masked dense attention; we report analytic FLOPs and make no wall-clock claims."
    

18\. Statistics and honesty rules
---------------------------------

*   A difference counts only if it exceeds three times the Phase 0 noise standard deviation (or its paired bootstrap interval excludes zero).
    
*   Report negative results and failed gates.
    
*   Do not tune thresholds after seeing results.
    
*   Mark every number as measured or estimated.
    

19\. FAQ (consult this for doubts)
----------------------------------

**Why not just use a bigger sliding window?** It is the strongest simple competitor, and we compare against it at equal memory. The project's claim is about how to choose what to keep, not that retention always beats a window.

**Why a shared router?** Because it would add no parameters and no extra stage if it works. Whether it works is the hypothesis.

**What is "hindsight"?** Using the future during training only. We measure which blocks the later tokens actually attended to and train the scorer to predict that from the prefix alone.

**Why synthetic data?** At 10M parameters, real text has weak recall signal. Synthetic regimes let us control when future demand is visible, and they have clean oracles.

**Why 10M parameters?** It fits free Kaggle T4s. Larger scale is future work.

**Why no speed claim?** Masked dense attention on a T4 at this size is not faster than dense attention. We report analytic FLOPs and memory.

**What is purge exactly?** At position P, delete all prefix KV except the window, sink tokens, and the m retained blocks. The SSM state is kept.

**Is memory constant after the switch?** No. It is bounded and tunable (state + window + sink + m blocks).

**Where does the SSM state come from?** The SSM scans the prefix too. The state at P is the real scan state. This replaces V7's converter and decay-injection trick.

**What is the oracle?** The best choice using future information. It is a ceiling, not a deployable method.

**What if a baseline wins?** Report it. The paper stays valid if the scorer comparison is rigorous.

**What if someone publishes the same thing?** Use the pivots in Section 14.3.

**What if everything ties?** Then the finding is that attention routing is sufficient, or the regimes need redesign. Either way you can report it as a study, but check Gate 2 first.

20\. Glossary
-------------

*   **KV cache:** stored keys and values of past tokens.
    
*   **SSM state:** fixed-size summary of the past.
    
*   **TransPoint P:** position where the model switches from attention to SSM.
    
*   **Prefix / suffix:** tokens before / after P.
    
*   **Block:** 32 consecutive tokens.
    
*   **Router:** small network scoring blocks for a query.
    
*   **Retained blocks (m):** prefix blocks kept after purge.
    
*   **Purge:** deleting old KV except what is retained.
    
*   **Sink tokens:** the first 4 tokens, always kept.
    
*   **Window:** the most recent W tokens, kept as a rolling buffer.
    
*   **Hindsight target:** attention mass that later tokens put on each block.
    
*   **Oracle:** ceiling policy using future information.
    
*   **LOBO:** leave-one-block-out loss increase.
    
*   **MQAR:** multi-query associative recall (key-value lookup test).
    
*   **Token-equivalent:** memory unit equal to 512 floats per layer (one cached token).
    
*   **S0 / S1 / S2:** the three router training configurations.
    
*   **RetentionBench:** our evaluation harness (regimes, policies, oracles, metrics).
    

21\. What to do next
--------------------

1.  Do the literature refresh and read ARM, HAM, laLTE in full (checkpoint 1).
    
2.  Run Phase 0 and record tokens/sec and the noise floor.
    
3.  Write the pre-registration file with hypotheses and gate thresholds.
    
4.  Build and test the small pieces (Phase 2) before touching real text.
    
5.  Run Phase 3. It decides whether the full project goes ahead.
    

If you send me the Phase 0 numbers, I will rescale Sections 8, 13 and 15 with real figures. I can also expand any section into line-by-line implementation notes (for example the chunked scan, the block mask builder, or the cache purge).