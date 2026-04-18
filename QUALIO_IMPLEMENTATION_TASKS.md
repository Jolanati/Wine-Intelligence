# Qualio — Implementation To-Do List

> Work top to bottom. Do not move to the next phase until all boxes in the current one are checked.
> Local = small dataset, CPU, fast feedback. Colab = full dataset, GPU, real training.

---

## Phase 0 — Setup
**Goal:** Environment ready, data on disk, first rows visible.
**Where:** Local machine
**Time:** ~2 hours

- [ ] Create project folder structure (`notebooks/`, `src/`, `deployment/`, `data/`, `outputs/`, `weights/`)
- [ ] Create and activate Python virtual environment
- [ ] Install all dependencies (`torch`, `torchvision`, `datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `nltk`, `pillow`, `fastapi`, `uvicorn`, `pytorch-grad-cam`, `wordcloud`)
- [ ] Save `requirements.txt`
- [ ] Download WineSensed `small` split from Hugging Face
- [ ] Save `images_reviews_attributes.csv` to `data/`
- [ ] Download and save `napping.csv` to `data/`
- [ ] Verify both files load and all expected columns are present
- [ ] Confirm at least one image URL is reachable from the `image` column

**Done when:** Both CSVs load without errors and you can see wine names, reviews, and image links.

---

## Phase 1 — Exploratory Data Analysis
**Goal:** Understand both datasets, spot class imbalance, confirm rubric minimums.
**Where:** Local machine
**Notebook:** `01_eda.ipynb`
**Time:** ~3 hours

### Image dataset

- [ ] Filter to 5 countries: France, Italy, Spain, US, Argentina
- [ ] Plot bar chart of label count per country — note which is dominant
- [ ] Download and display a 4x4 sample grid of label images, one row per country
- [ ] Report image dimensions and confirm all are consistent
- [ ] Note any download failures or corrupt images

### Text dataset

- [ ] Derive `tier` column from `rating` column (below 3.5 = Entry, 3.5-4.0 = Premium, above 4.0 = Exceptional)
- [ ] Plot bar chart of quality tier distribution — note imbalance
- [ ] Calculate and report average review length in words
- [ ] Confirm average exceeds 15 words (rubric minimum)
- [ ] Plot review length histogram
- [ ] Generate one word cloud per quality tier
- [ ] Display 3 representative reviews per tier side by side
- [ ] Report number of null/empty reviews and decide whether to drop or fill

### Napping data

- [ ] Load `napping.csv` and inspect columns
- [ ] Plot all wines as dots using `coor1` and `coor2` coordinates
- [ ] Count how many wines from `images_reviews_attributes.csv` appear in napping (join on `experiment_id`)
- [ ] Note the 108-wine subset — record which vintages are covered

**Done when:** You have 5 saved chart images in `outputs/` and understand the shape of both datasets.

---

## Phase 2 — Preprocessing
**Goal:** Both datasets are clean, encoded, and ready for PyTorch dataloaders.
**Where:** Local machine
**Notebook:** `02_preprocessing.ipynb`
**Time:** ~3 hours

### Image preprocessing

- [ ] Define train transforms: resize to 64x64 locally (224x224 on Colab), random flip, rotation ±15°, colour jitter
- [ ] Define val/test transforms: resize only, no augmentation
- [ ] Apply ImageNet mean/std normalisation (required for ResNet-18)
- [ ] Write `WineImageDataset` class that returns (image tensor, country label)
- [ ] Verify one batch loads without errors and tensor shapes are correct
- [ ] Confirm labels are integers 0-4 mapped to the 5 countries

### Text preprocessing

- [ ] Drop rows where `review` is null
- [ ] Lowercase all review text
- [ ] Tokenise using `nltk.word_tokenize`
- [ ] Build vocabulary from training set only, capped at 20,000 words
- [ ] Add `<PAD>` (index 0) and `<UNK>` (index 1) tokens
- [ ] Encode each review as a list of integer IDs
- [ ] Pad short reviews and truncate long reviews to 200 tokens
- [ ] Report what percentage of reviews get truncated
- [ ] Download GloVe-100d embeddings
- [ ] Build GloVe embedding matrix aligned to your vocabulary
- [ ] Report vocabulary coverage percentage (how many words found in GloVe)
- [ ] Write `WineReviewDataset` class that returns (token tensor, tier label)
- [ ] Verify one batch loads without errors

**Done when:** Both datasets return correct tensor shapes from a DataLoader.

---

## Phase 3 — Train / Val / Test Split
**Goal:** Three clean splits per dataset, balanced, sealed test sets.
**Where:** Local machine
**Notebook:** `02_preprocessing.ipynb` (continued)
**Time:** ~30 minutes

- [ ] Split image dataset: 70% train, 15% val, 15% test — stratified by country, `random_state=42`
- [ ] Split text dataset: 70% train, 15% val, 15% test — stratified by tier, `random_state=42`
- [ ] Print class balance table for each split of each dataset — confirm stratification worked
- [ ] Save the split index files (or split CSVs) so splits are reproducible
- [ ] Do not touch test sets again until final evaluation — note this in the notebook

**Done when:** Six splits exist (train/val/test × image/text), all balanced, all saved.

---

## Phase 4 — CNN Branch (Local prototype)
**Goal:** End-to-end CNN pipeline working on small data. Accuracy does not matter yet.
**Where:** Local machine
**Notebook:** `03_cnn.ipynb`
**Time:** ~2 hours (local) + 6-8 hours training on Colab

### Scratch CNN — local

- [ ] Build a 3-block custom CNN (Conv → BatchNorm → ReLU → MaxPool per block, global average pool, dropout, dense head)
- [ ] Print model summary and count trainable parameters
- [ ] Run one forward pass with a dummy batch — confirm output shape is `[batch, 5]`
- [ ] Train for 2 epochs on 300 local samples — confirm loss decreases
- [ ] Plot training loss curve
- [ ] Save model weights to `weights/cnn_scratch_local.pt`
- [ ] Load weights back and confirm they restore correctly

### ResNet-18 — local

- [ ] Load ResNet-18 with ImageNet pretrained weights
- [ ] Freeze all backbone layers
- [ ] Replace the final `fc` layer with dropout + linear head for 5 classes
- [ ] Count trainable vs frozen parameters — confirm only head is trainable
- [ ] Run one forward pass with a dummy batch — confirm output shape is `[batch, 5]`
- [ ] Train for 2 epochs on 300 local samples — confirm loss decreases
- [ ] Save weights to `weights/cnn_resnet_local.pt`
- [ ] Load weights back and confirm they restore correctly

**Done when:** Both CNNs train without errors locally and weights save/load cleanly.

---

## Phase 5 — CNN Branch (Colab full training)
**Where:** Google Colab — T4 GPU
**Time:** ~1-2 hours per model

### Colab setup

- [ ] Open new Colab notebook, set runtime to T4 GPU
- [ ] Mount Google Drive, create `qualio/` folder
- [ ] Install dependencies
- [ ] Upload or download the full WineSensed dataset
- [ ] Resize images to 224x224 for this run
- [ ] Sample up to 2,000 images per country (10,000 total) for balanced training

### Scratch CNN — full training

- [ ] Train for 20 epochs with early stopping (patience = 5)
- [ ] Save best checkpoint to Drive after each epoch
- [ ] Plot training and validation loss + accuracy curves
- [ ] Evaluate on test set: report accuracy, F1-score (macro), confusion matrix
- [ ] Save final weights to Drive as `cnn_scratch_best.pt`

### ResNet-18 — full training

- [ ] Phase 1: train frozen backbone for 10 epochs, only head learns
- [ ] Phase 2: unfreeze last 2 conv blocks (`layer3`, `layer4`), lower learning rate, train 10 more epochs
- [ ] Save best checkpoint per phase
- [ ] Plot training and validation loss + accuracy curves for both phases
- [ ] Evaluate on test set: report accuracy, F1-score (macro), confusion matrix
- [ ] Compare scratch CNN vs ResNet-18 results in a summary table
- [ ] Write 1-2 paragraphs explaining which performed better and why
- [ ] Save final weights to Drive as `cnn_resnet_best.pt`
- [ ] Download both weight files to local `weights/` folder

**Done when:** Both CNNs have test metrics, comparison table written, weights saved locally.

---

## Phase 6 — BiLSTM Branch (Local prototype)
**Goal:** End-to-end LSTM pipeline working on small text data.
**Where:** Local machine
**Notebook:** `04_bilstm.ipynb`
**Time:** ~2 hours (local) + 4-5 hours training on Colab

### Unidirectional LSTM — local

- [ ] Build LSTM: Embedding → LSTM (1 layer, 256 hidden) → dropout → linear head (3 classes)
- [ ] Load GloVe matrix into the embedding layer
- [ ] Run one forward pass with a dummy batch — confirm output shape is `[batch, 3]`
- [ ] Train for 2 epochs on 500 local reviews — confirm loss decreases
- [ ] Save weights to `weights/lstm_local.pt`

### Bidirectional LSTM — local

- [ ] Build BiLSTM: same as above but `bidirectional=True`, concatenate forward + backward final hidden states
- [ ] Confirm output shape is `[batch, 3]`
- [ ] Train for 2 epochs on 500 local reviews — confirm loss decreases
- [ ] Save weights to `weights/bilstm_local.pt`

**Done when:** Both LSTM variants train without errors locally.

---

## Phase 7 — BiLSTM Branch (Colab full training)
**Where:** Google Colab — T4 GPU
**Time:** ~1-2 hours

- [ ] Sample up to 5,000 reviews per tier (15,000 total) for balanced training
- [ ] Train unidirectional LSTM for 15 epochs with early stopping
- [ ] Train BiLSTM for 15 epochs with early stopping
- [ ] Plot training and validation loss + accuracy curves for both
- [ ] Evaluate both on test set: accuracy, F1-score (macro), confusion matrix
- [ ] Compare both variants in a summary table
- [ ] Write paragraph explaining why BiLSTM outperforms unidirectional (reads context in both directions)
- [ ] Run BiLSTM on the entire dataset — predict quality tier for every review
- [ ] Save enriched dataframe (original columns + `predicted_tier` + `tier_confidence`) to `enriched_dataset.csv`
- [ ] Download `enriched_dataset.csv` and `bilstm_best.pt` to local machine

**Done when:** Both LSTM variants have test metrics, `enriched_dataset.csv` is saved locally.

---

## Phase 8 — Explainability
**Goal:** Visual evidence of what each model actually learned.
**Where:** Local machine (use small images and a few reviews)
**Notebook:** `06_explainability.ipynb`
**Time:** ~2-3 hours

### Grad-CAM for CNN

- [ ] Install `pytorch-grad-cam` if not already done
- [ ] Select 6 test images — one or two per country
- [ ] Run Grad-CAM targeting the last convolutional layer of ResNet-18
- [ ] Overlay heatmap on original label image
- [ ] Save 6 Grad-CAM images to `outputs/`
- [ ] Write a short observation: what part of the label did the model attend to? (typography, colour, shape, logo placement)

### Attention / SHAP for BiLSTM

- [ ] Select 6 test reviews — two per quality tier
- [ ] Extract per-word attention weights from the LSTM hidden states
- [ ] Visualise word importance as a colour-coded highlight (darker = higher weight)
- [ ] Save 6 annotated review visualisations to `outputs/`
- [ ] Write a short observation: which words drove the Exceptional prediction? Which drove Entry?

**Done when:** 12 explainability visualisations saved, each with a one-line interpretation written in the notebook.

---

## Phase 9 — Business Integration
**Goal:** Combine both model outputs into the sourcing brief. Produce the 20-example table required by the rubric.
**Where:** Local machine
**Notebook:** `05_integration.ipynb`
**Time:** ~3 hours

### Napping similarity

- [ ] Load `napping.csv`
- [ ] For each wine, average napping coordinates across all participants who tasted it
- [ ] Write function: given a `vintage_id`, return the 3 nearest wines by Euclidean distance in napping space
- [ ] Convert distance to a similarity percentage (0-100)
- [ ] Write fallback function for wines not in the napping subset: match by primary grape and price range
- [ ] Test on 5 vintages — confirm results look sensible

### Sourcing brief logic

- [ ] Write `get_recommendation(country, tier)` function that returns a plain-language stocking recommendation
- [ ] Cover all 15 combinations (5 countries × 3 tiers)
- [ ] Test on 10 examples — confirm output reads naturally

### 20-example integration table (rubric requirement)

- [ ] Sample 20 wines from the test set
- [ ] For each: run CNN prediction, retrieve BiLSTM tier from enriched CSV, retrieve wine attributes, generate recommendation
- [ ] Build DataFrame with columns: wine name, CNN prediction, BiLSTM tier, actual rating, price, combined recommendation
- [ ] Save as `outputs/integration_examples.csv`
- [ ] Save as formatted table in the notebook

### Comparison chart (rubric requirement)

- [ ] Plot three bars: CNN test accuracy, BiLSTM test accuracy, business integration coverage (% of test set with full combined output)
- [ ] Save as `outputs/comparison_chart.png`

**Done when:** 20-example table exists, comparison chart saved, recommendation function covers all combinations.

---

## Phase 10 — Deployment
**Goal:** Working FastAPI app serving the Qualio frontend locally. Then deploy to Render.
**Where:** Local → Render
**Time:** ~4-5 hours

### Local FastAPI

- [ ] Create `deployment/app.py` with FastAPI application
- [ ] Add `/predict` endpoint that accepts an uploaded image
- [ ] Load CNN weights at startup
- [ ] Load `enriched_dataset.csv` and `napping.csv` at startup
- [ ] Run CNN on uploaded image — return country prediction + confidence
- [ ] Look up matched vintage in enriched CSV — return all wine attributes
- [ ] Run napping similarity function — return 3 similar wines
- [ ] Return full JSON response
- [ ] Copy `qualio-v4.html` into `deployment/static/index.html`
- [ ] Mount static files so the HTML is served at root
- [ ] Run locally: `uvicorn app:app --reload --port 8000`
- [ ] Open `localhost:8000` — confirm Qualio loads
- [ ] Upload a test label image — confirm the full sourcing brief appears
- [ ] Take a screenshot of the working prototype — save to `outputs/prototype_screenshot.png`

### Render deployment

- [ ] Create `deployment/requirements.txt` with all production dependencies
- [ ] Create a GitHub repository and push the `deployment/` folder
- [ ] Create new Render web service, connect to the GitHub repo
- [ ] Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- [ ] Wait for build to complete — fix any dependency errors
- [ ] Open the Render URL — confirm Qualio loads
- [ ] Test with a real label image — confirm full output appears
- [ ] Copy the public Render URL — this is your submission deployment link

**Done when:** Qualio is live on a public URL and returns a full sourcing brief for an uploaded label.

---

## Phase 11 — Notebook cleanup and PDF export
**Goal:** Single clean submission notebook that runs end to end without errors.
**Time:** ~3-4 hours

- [ ] Merge all notebook sections into `Final_Project.ipynb` in rubric order (EDA → preprocessing → CNN → RNN → business integration → explainability → deployment)
- [ ] Add markdown commentary between every code cell — explain what and why, not just what
- [ ] Add contribution table at the top (solo: all sections by you)
- [ ] Add References section at the end — cite WineSensed paper, GloVe, ResNet-18 paper, pytorch-grad-cam
- [ ] Restart kernel and run all cells top to bottom — fix any errors
- [ ] Confirm all 12 explainability images display inline
- [ ] Confirm the 20-example table renders as a formatted table
- [ ] Confirm all training curves display for all 4 models (2 CNN, 2 LSTM)
- [ ] Export as PDF: File → Download → Download as PDF
- [ ] Open the PDF and verify all outputs, plots, and markdown cells are visible
- [ ] Confirm no cell outputs are cut off or missing

**Done when:** PDF opens cleanly with all sections visible and complete.

---

## Phase 12 — Presentation
**Goal:** 15-20 minute slide deck ready for the final lecture.
**Time:** ~4 hours

### Slides to build (in order)

- [ ] Title slide — project name, your name, date
- [ ] Business problem — who is the user, what problem does Qualio solve, why does it need both image and text analysis
- [ ] Dataset overview — WineSensed in 3 bullet points, key stats, napping experiment explained in one sentence
- [ ] EDA highlights — 2-3 visuals: country distribution, tier distribution, one word cloud comparison
- [ ] Architecture diagram — CNN branch and BiLSTM branch as two parallel paths, meeting at the business layer
- [ ] CNN results — scratch vs ResNet-18 training curves, test metrics table, one Grad-CAM example
- [ ] BiLSTM results — LSTM vs BiLSTM training curves, test metrics table, one attention example
- [ ] Business integration — the 20-example table (top 6 rows), sourcing recommendation examples
- [ ] Similar wines — napping experiment explained visually, the taste space map
- [ ] Live demo — open Qualio in browser, upload a label, walk through the output live
- [ ] Ethics and limitations — review bias, label aesthetics vs actual origin, price data staleness, napping subset coverage
- [ ] What I would do next — joint model bonus, larger image subset, real-time label OCR for vintage matching
- [ ] References slide

### Preparation

- [ ] Rehearse the demo section with a real label image before the presentation
- [ ] Record a 3-minute screen recording of the demo as backup in case of live connection issues
- [ ] Upload screen recording to Drive and add the link to your Moodle submission

**Done when:** All 13 slides exist, demo recorded, everything uploaded to Moodle.

---

## Final Submission Checklist

- [ ] `Final_Project.ipynb` — runs end to end without errors
- [ ] `Final_Project.pdf` — all outputs visible
- [ ] Presentation `.pptx` or `.pdf` — 13+ slides
- [ ] Deployment URL — Qualio live on Render
- [ ] Screen recording URL — 3-minute demo backup
- [ ] All submitted via Moodle before the deadline

---

## Time Summary

| Phase | Where | Estimate |
|---|---|---|
| 0 — Setup | Local | 2 hrs |
| 1 — EDA | Local | 3 hrs |
| 2-3 — Preprocessing + splits | Local | 3.5 hrs |
| 4 — CNN local prototype | Local | 2 hrs |
| 5 — CNN full training | Colab | 2 hrs |
| 6 — BiLSTM local prototype | Local | 2 hrs |
| 7 — BiLSTM full training | Colab | 2 hrs |
| 8 — Explainability | Local | 3 hrs |
| 9 — Business integration | Local | 3 hrs |
| 10 — Deployment | Local + Render | 5 hrs |
| 11 — Notebook cleanup | Local | 4 hrs |
| 12 — Presentation | Local | 4 hrs |
| **Total** | | **~35 hours** |

---

*Tip: phases 0-3 and the local prototypes (4, 6) can be done in week 1.
Full Colab training (5, 7) in week 2. Everything else in week 3.*
