# QR-Bloom — common task targets
#
# One DiT3D model is trained across all QR versions (v2..v5). Checkpoints
# and per-epoch artifacts live in:
#   checkpoints/qrbloom_all.pt          latest snapshot
#   checkpoints/qrbloom_all_best.pt     lowest validation loss so far
#   runs_all/                           per-epoch JSON + montage + log
#
# ─────────────────────────────────────────────────────────────────────────
# Quick reference
# ─────────────────────────────────────────────────────────────────────────
#
#   make help                    Print all targets with a one-liner each.
#
#   make train                   Launch training in the background (nohup);
#                                resumes from checkpoints/qrbloom_all.pt.
#   make stop                    Kill the running training process.
#   make status                  GPU usage + latest epoch / val per version.
#   make logs                    Tail -f the training log.
#
#   make export                  Export EMA weights for the browser demo
#                                into docs/assets/.
#
#   make gallery                 Start the FastAPI viewer (CPU inference,
#                                safe to run alongside training).
#                                Override: PORT=9000 make gallery
#   make gallery-gpu             Same but inference on the freest GPU.
#   make stop-gallery            Kill the gallery server.
#
#   make clean-checkpoints       Move checkpoints/qrbloom*.pt → backups/<ts>/.
#   make clean-runs              Move runs_all/ → backups/<ts>/.
#
# ─────────────────────────────────────────────────────────────────────────
# Customization
# ─────────────────────────────────────────────────────────────────────────
#
# Training knobs are Make variables with defaults tuned for a single ~8GB
# GPU. Override at the command line, e.g.:
#
#   BATCH=32 EPOCHS=500 make train
#   QR_VERSIONS=2,3 make train
#
# ─────────────────────────────────────────────────────────────────────────

# Python with torch installed — override for your environment:
#   PY=/path/to/venv/bin/python make train
PY      ?= python3
TS      := $(shell date +%Y%m%d_%H%M%S)
BAK_DIR := backups/$(TS)
PORT    ?= 8000

# Training config — BATCH is sized for ~8GB VRAM at the largest grid (v5).
EPOCHS         ?= 300
EPOCH_SIZE     ?= 40000
VAL_SIZE       ?= 648
BATCH          ?= 18
LR             ?= 1e-4
SAMPLE_STEPS   ?= 80
WORKERS        ?= 4
QR_VERSIONS    ?= 2,3,4,5
MONT_VERSION   ?= 2

TRAIN_ENV := CUDA_DEVICE_ORDER=PCI_BUS_ID \
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
             VARIANT=_all QR_VERSIONS=$(QR_VERSIONS) QR_VERSIONS_VAL=$(QR_VERSIONS) \
             BATCH=$(BATCH) LR=$(LR) EPOCHS=$(EPOCHS) EPOCH_SIZE=$(EPOCH_SIZE) \
             VAL_SIZE=$(VAL_SIZE) SAMPLE_STEPS=$(SAMPLE_STEPS) WORKERS=$(WORKERS) \
             MONT_VERSION=$(MONT_VERSION)

.PHONY: help train stop status logs export \
        gallery gallery-gpu stop-gallery \
        clean-checkpoints clean-runs

help: ## Show this help.
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

# ── Training ─────────────────────────────────────────────────────────────

train: ## Launch training in the background (resumes if a checkpoint exists).
	@mkdir -p runs_all
	@$(TRAIN_ENV) nohup $(PY) train.py >> runs_all/train.log 2>&1 &
	@sleep 1 && echo "training launched  →  tail -f runs_all/train.log"

stop: ## Kill the training process.
	@pkill -KILL -f "python.*train\.py" 2>/dev/null || true
	@sleep 2 && echo "training killed"

status: ## GPU usage + latest val metrics.
	@echo "── GPU ──────────────────────────────────────"
	@CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi \
	  --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
	  --format=csv,noheader
	@echo ""
	@echo "── Latest epoch ─────────────────────────────"
	@grep "val_total" runs_all/train.log 2>/dev/null | tail -3 || echo "(no log)"

logs: ## Follow the training log.
	@tail -f runs_all/train.log

# ── Export ───────────────────────────────────────────────────────────────

export: ## Export EMA weights + manifest for the browser demo.
	@$(PY) scripts/export_for_tfjs.py

# ── Gallery ──────────────────────────────────────────────────────────────

gallery: ## Start the FastAPI gallery on :$(PORT) (CPU inference).
	@GALLERY_DEVICE=cpu nohup $(PY) gallery.py --port $(PORT) > /tmp/qrbloom-gallery.log 2>&1 &
	@sleep 2 && echo "gallery → http://localhost:$(PORT)/  (log: /tmp/qrbloom-gallery.log)"

gallery-gpu: ## Start the gallery on :$(PORT) picking the freest GPU.
	@nohup $(PY) gallery.py --port $(PORT) > /tmp/qrbloom-gallery.log 2>&1 &
	@sleep 2 && echo "gallery (GPU) → http://localhost:$(PORT)/"

stop-gallery: ## Kill the gallery server.
	@pkill -KILL -f "python.*gallery\.py" 2>/dev/null || true
	@echo "gallery stopped"

# ── Cleanup ──────────────────────────────────────────────────────────────

clean-checkpoints: ## Move checkpoints/qrbloom*.pt → backups/<ts>/.
	@mkdir -p $(BAK_DIR)
	@mv checkpoints/qrbloom*.pt $(BAK_DIR)/ 2>/dev/null || true
	@echo "checkpoints moved to $(BAK_DIR)/"

clean-runs: ## Move runs_all/ → backups/<ts>/.
	@mkdir -p $(BAK_DIR)
	@mv runs_all $(BAK_DIR)/ 2>/dev/null || true
	@echo "runs_all moved to $(BAK_DIR)/"
