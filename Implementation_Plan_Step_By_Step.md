# Qualio — Implementation Plan Step by Step

---

## Phase 1 — Data (Day 1–2)

- [ ] Download WineSensed `small` split locally via `load_dataset('Dakhoo/L2T-NeurIPS-2023', 'small')`
- [ ] Save `images_reviews_attributes.csv` and `napping.csv` to `data/`
- [ ] Open `01_eda.ipynb` and run country distribution bar chart
- [ ] Display 4×4 image grid with one row per country
- [ ] Report image dimensions (H×W×C) and channel mean/std statistics *(rubric requirement)*
- [ ] Report total number of text samples in the dataset *(rubric requirement)*
- [ ] Plot review length histogram — confirm average > 15 words
- [ ] Generate word clouds per quality tier (Entry / Premium / Exceptional)
- [ ] Print top-20 most frequent terms per quality tier as a ranked list *(rubric requirement)*
- [ ] Print 3 raw review samples per tier *(rubric requirement)*
- [ ] Check `napping.csv` — count unique `experiment_id` values and confirm join to `vintage_id`

---

## Phase 2 — Preprocessing (Day 2–3)

- [ ] Build `train_transforms`: resize 224×224, random flip, rotation, color jitter, ImageNet normalise
- [ ] Write prose explanation of each augmentation choice and why it suits wine labels *(rubric requirement)*
- [ ] Build `val_transforms`: resize + normalise only
- [ ] Build `WineImageDataset` — loads image from path, returns tensor + country label int
- [ ] Verify one batch loads without errors
- [ ] Build `build_vocab()` on training reviews only (no data leakage)
- [ ] Build `encode()` with pad/truncate at MAX\_LEN=200 — report % truncated
- [ ] Download GloVe-100d, build embedding matrix, report vocabulary coverage %
- [ ] Write prose justification for word-level tokenisation choice over subword/character *(rubric requirement)*
- [ ] Build `WineTextDataset`
- [ ] CNN split: 70/15/15 stratified by country, seed=42
- [ ] RNN split: 70/15/15 stratified by quality tier, seed=42
- [ ] Seal both test sets — do not touch until final evaluation

---

## Phase 3 — CNN (Day 3–5)

- [ ] Implement `ScratchCNN` — 3 conv blocks, global avg pool, dropout, linear head
- [ ] Print and record number of trainable parameters for `ScratchCNN` *(rubric requirement)*
- [ ] Train `ScratchCNN` locally on small data to verify shapes (5 epochs)
- [ ] Move to Colab T4, load full dataset, train `ScratchCNN` for 30 epochs
- [ ] Plot train/val loss and accuracy curves for `ScratchCNN`
- [ ] Implement `build_resnet18()` — frozen backbone, replace head with dropout + linear(5)
- [ ] Train ResNet-18 head only for 5 epochs on Colab
- [ ] Unfreeze `layer4` + `layer3`, reduce LR to 1e-4, train 10 more epochs
- [ ] Plot train/val loss and accuracy curves for ResNet-18
- [ ] Evaluate both models on test set — Accuracy, weighted F1, Confusion Matrix
- [ ] Save best ResNet-18 weights to `weights/cnn_resnet18_best.pt`
- [ ] Write comparison discussion: which approach wins and why

---

## Phase 4 — BiLSTM (Day 5–6)

- [ ] Implement `WineLSTM` with `bidirectional=False` (Variation 1 — unidirectional LSTM)
- [ ] Implement `WineLSTM` with `bidirectional=True` (Variation 2 — BiLSTM)
- [ ] Train both variations on Colab, plot **train and val** loss and accuracy curves for each *(rubric requirement)*
- [ ] Evaluate both on test set — Accuracy, weighted F1, Confusion Matrix
- [ ] Save best BiLSTM weights to `weights/bilstm_best.pt`
- [ ] **Bonus +3 pts:** Fine-tune DistilBERT on quality tier classification, compare vs BiLSTM on test set
- [ ] Run `batch_predict_tiers()` on full dataset using the best BiLSTM model
- [ ] Save output as `data/enriched_dataset.csv` with `predicted_tier` and `tier_confidence` columns

---

## Phase 5 — Business Integration (Day 7)

- [ ] Implement `get_similar_wines()` using Euclidean distance on napping coordinates
- [ ] Implement `_fallback_similar()` for wines outside the 108-bottle napping subset
- [ ] Implement `get_recommendation()` — country × tier decision matrix
- [ ] Sample 20 rows and build integration table: CNN prediction + BiLSTM tier + combined recommendation
- [ ] Save table to `outputs/integration_examples.csv`
- [ ] Write narrative analysis: identify 3–5 examples where the combined output gives a better or more complete insight than either model alone *(rubric requirement)*
- [ ] Build comparison bar chart: CNN test accuracy vs BiLSTM test accuracy vs combined quality *(rubric requirement)*

---

## Phase 6 — Explainability (Day 7–8)

- [ ] Run Grad-CAM on 5 ResNet-18 test images (one per country) — save visualisations
- [ ] For each Grad-CAM image, write 2–3 sentences explaining what region the model attends to
- [ ] Extract BiLSTM attention weights for one review per quality tier
- [ ] Display top 10 attended words per review with bar visualisation
- [ ] Write 2–3 sentences per example explaining what the model focuses on

---

## Phase 7 — Deployment (Day 8–9)

- [ ] Add `/predict-text` POST endpoint to `app.py` — accepts review text, returns BiLSTM tier + confidence
- [ ] Add text area input to `index.html` wired to `/predict-text`
- [ ] Test image upload → `/predict` → CNN country + wine fact card + similar wines
- [ ] Test text input → `/predict-text` → quality tier + confidence score
- [ ] Verify combined badge renders: e.g. "Exceptional · French Red"
- [ ] Create `deployment/requirements.txt`
- [ ] Deploy to Render — start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] Upload `enriched_dataset.csv`, `napping.csv`, both `.pt` weight files to Render
- [ ] Take screenshot of working prototype for notebook deliverable

---

## Phase 8 — Final Notebook (Day 9–10)

- [ ] Create `Final_Project.ipynb`
- [ ] Add contribution table (your name against all components)
- [ ] Copy in EDA highlights from `01_eda.ipynb`
- [ ] Copy in preprocessing decisions from `02_preprocessing.ipynb`
- [ ] Add CNN comparison section: curves + test metrics for both approaches
- [ ] Add BiLSTM comparison section: curves + test metrics for both variations
- [ ] Add business integration section: 20-example table + comparison chart
- [ ] Add explainability section: Grad-CAM images + attention word lists
- [ ] Add deployment section: screenshot + live link
- [ ] Add business framing + ethics section
- [ ] Add References section
- [ ] Add dataset download script/cell so notebook can be reproduced without manual downloads *(rubric requirement)*
- [ ] Verify both `.pt` weight files load correctly with a test inference cell *(rubric requirement)*
- [ ] Export notebook to PDF — verify all plots are visible

---

## Phase 9 — Presentation (Day 10)

- [ ] Slide 1 — Title, your name, date
- [ ] Slide 2 — Business problem: who is the wine buyer, what pain do they have
- [ ] Slide 3 — Dataset overview: WineSensed stats + napping experiment diagram
- [ ] Slide 4 — Architecture diagram: CNN path + BiLSTM path + integration layer
- [ ] Slide 5 — CNN results: scratch vs ResNet-18 curves + confusion matrices
- [ ] Slide 6 — BiLSTM results: unidirectional vs BiLSTM curves + confusion matrices
- [ ] Slide 7 — Business integration: 20-example table + comparison chart
- [ ] Slide 8 — Explainability: 2 Grad-CAM images + 1 attention example
- [ ] Slide 9 — Live demo or screen recording of Qualio
- [ ] Slide 10 — Ethics: label bias, review language bias, price data staleness
- [ ] Slide 11 — Conclusions + what to explore next
- [ ] Slide 12 — Contributions + references

---

## Bonus — Joint Model (+10 pts, only if everything above is done)

- [ ] Extract CNN feature vector from global avg pool layer (before classification head)
- [ ] Extract BiLSTM feature vector from final hidden state
- [ ] Concatenate both vectors, pass through 2 FC layers, predict quality tier
- [ ] Train joint model on samples that have both image and review
- [ ] Compare: CNN-only vs BiLSTM-only vs Joint model — accuracy + F1
- [ ] Add joint model section to `Final_Project.ipynb`

---

## Priority Order If Time Runs Short

1. [ ] ResNet-18 trained + weights saved
2. [ ] BiLSTM trained + enriched CSV saved
3. [ ] Text input working in deployment frontend
4. [ ] 20-example table + comparison chart in notebook
5. [ ] Grad-CAM + attention explainability
6. [ ] Joint model bonus
