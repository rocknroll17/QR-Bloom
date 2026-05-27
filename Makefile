# QR-Bloom — common task targets
#
# QR-Bloom trains one separate UNet3D model per QR version (v2..v5). Each
# version's checkpoint and per-epoch artifacts go in version-scoped paths:
#   checkpoints/qrbloom_v{N}.pt          latest snapshot
#   checkpoints/qrbloom_v{N}_best.pt     lowest validation loss so far
#   runs_v{N}/                           per-epoch JSON + log
#
# ─────────────────────────────────────────────────────────────────────────
# Quick reference
# ─────────────────────────────────────────────────────────────────────────
#
#   make help                    Print all targets with a one-liner each.
#
#   make train                   Launch all 4 per-version trainings.
#                                Each runs in nohup; logs in runs_v{N}/train.log.
#   make train-v{2..5}           Launch one specific version.
#
#   make stop                    Kill all running train.py processes.
#   make stop-v{2..5}            Kill one version's training.
#
#   make status                  GPU usage + which trainings are alive +
#                                latest epoch / val_total per version.
#   make logs-v{2..5}            Tail -f a single training's log.
#
#   make gallery                 Start the FastAPI viewer (CPU inference,
#                                safe to run alongside training).
#                                Override: PORT=9000 make gallery
#   make gallery-gpu             Same but inference on the first free GPU.
#   make stop-gallery            Kill the gallery server.
#
#   make clean-checkpoints       Move all qrbloom*.pt → backups/<timestamp>/.
#   make clean-runs              Move runs_v{N}/ → backups/<timestamp>/.
#
# ─────────────────────────────────────────────────────────────────────────
# Customization
# ─────────────────────────────────────────────────────────────────────────
#
# All per-version knobs (GPU index, BATCH, EPOCHS, EPOCH_SIZE, ...) are
# Make variables with sensible defaults. Override them at the command line:
#
#   V2_GPU=0 V2_BATCH=64 make train-v2
#   EPOCHS=500 EPOCH_SIZE=100000 make train
#
# The default per-version BATCH values are tuned for ~24GB GPUs. If you
# have less (or more) VRAM, lower (or raise) V{N}_BATCH accordingly.
#
# ─────────────────────────────────────────────────────────────────────────

PY      := ./.venv/bin/python
TS      := $(shell date +%Y%m%d_%H%M%S)
BAK_DIR := backups/$(TS)
PORT    ?= 8000

# Shared training config — same for all versions unless overridden.
EPOCHS         ?= 300
EPOCH_SIZE     ?= 80000
VAL_SIZE       ?= 400
SAMPLE_STEPS   ?= 80
WORKERS        ?= 4

# Per-version GPU assignment. Defaults map v{2..5} onto CUDA_VISIBLE_DEVICES
# 2, 1, 3, 0 — adjust V{N}_GPU if your GPU layout differs.
V2_GPU ?= 2
V3_GPU ?= 1
V4_GPU ?= 3
V5_GPU ?= 0

# Per-version BATCH — tuned for ~24GB VRAM. Smaller QR versions fit more
# samples per batch. Lower these if you OOM, raise them if you have headroom.
V2_BATCH ?= 128
V3_BATCH ?= 80
V4_BATCH ?= 80
V5_BATCH ?= 56

COMMON_ENV := CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SHARED_VARS := EPOCHS=$(EPOCHS) EPOCH_SIZE=$(EPOCH_SIZE) VAL_SIZE=$(VAL_SIZE) \
               SAMPLE_STEPS=$(SAMPLE_STEPS) WORKERS=$(WORKERS) \
               QR_VERSION_WEIGHTS=1.0

V2_ENV := $(COMMON_ENV) CUDA_VISIBLE_DEVICES=$(V2_GPU) VARIANT=_v2 \
          BATCH=$(V2_BATCH) QR_VERSIONS=2 QR_VERSIONS_VAL=2 $(SHARED_VARS)

V3_ENV := $(COMMON_ENV) CUDA_VISIBLE_DEVICES=$(V3_GPU) VARIANT=_v3 \
          BATCH=$(V3_BATCH) QR_VERSIONS=3 QR_VERSIONS_VAL=3 $(SHARED_VARS)

V4_ENV := $(COMMON_ENV) CUDA_VISIBLE_DEVICES=$(V4_GPU) VARIANT=_v4 \
          BATCH=$(V4_BATCH) QR_VERSIONS=4 QR_VERSIONS_VAL=4 $(SHARED_VARS)

V5_ENV := $(COMMON_ENV) CUDA_VISIBLE_DEVICES=$(V5_GPU) VARIANT=_v5 \
          BATCH=$(V5_BATCH) QR_VERSIONS=5 QR_VERSIONS_VAL=5 $(SHARED_VARS)

.PHONY: help train train-v2 train-v3 train-v4 train-v5 \
        stop stop-v2 stop-v3 stop-v4 stop-v5 \
        gallery gallery-gpu stop-gallery \
        status logs-v2 logs-v3 logs-v4 logs-v5 \
        clean-checkpoints clean-runs

help: ## Show this help.
	@grep -E '^[a-zA-Z][a-zA-Z0-9_-]*:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

# ── Training ─────────────────────────────────────────────────────────────

train: train-v2 train-v3 train-v4 train-v5 ## Launch all 4 trainings in parallel.

train-v2: ## Train v2 only.
	@mkdir -p runs_v2
	@$(V2_ENV) nohup $(PY) train.py > runs_v2/train.log 2>&1 &
	@sleep 1 && echo "v2 launched  →  tail -f runs_v2/train.log"

train-v3: ## Train v3 only.
	@mkdir -p runs_v3
	@$(V3_ENV) nohup $(PY) train.py > runs_v3/train.log 2>&1 &
	@sleep 1 && echo "v3 launched  →  tail -f runs_v3/train.log"

train-v4: ## Train v4 only.
	@mkdir -p runs_v4
	@$(V4_ENV) nohup $(PY) train.py > runs_v4/train.log 2>&1 &
	@sleep 1 && echo "v4 launched  →  tail -f runs_v4/train.log"

train-v5: ## Train v5 only.
	@mkdir -p runs_v5
	@$(V5_ENV) nohup $(PY) train.py > runs_v5/train.log 2>&1 &
	@sleep 1 && echo "v5 launched  →  tail -f runs_v5/train.log"

# ── Stopping ─────────────────────────────────────────────────────────────

stop: ## Kill all train.py processes.
	@pkill -KILL -f "python.*train.py" 2>/dev/null || true
	@sleep 2 && echo "all trainings killed"

stop-v2: ## Kill v2 training.
	@pgrep -af "VARIANT=_v2.*train.py" | awk '{print $$1}' | xargs -r kill -KILL
	@echo "v2 killed"

stop-v3: ## Kill v3 training.
	@pgrep -af "VARIANT=_v3.*train.py" | awk '{print $$1}' | xargs -r kill -KILL
	@echo "v3 killed"

stop-v4: ## Kill v4 training.
	@pgrep -af "VARIANT=_v4.*train.py" | awk '{print $$1}' | xargs -r kill -KILL
	@echo "v4 killed"

stop-v5: ## Kill v5 training.
	@pgrep -af "VARIANT=_v5.*train.py" | awk '{print $$1}' | xargs -r kill -KILL
	@echo "v5 killed"

# ── Gallery ──────────────────────────────────────────────────────────────

gallery: ## Start the FastAPI gallery on :$(PORT) (CPU inference).
	@GALLERY_DEVICE=cpu nohup $(PY) gallery.py --port $(PORT) > /tmp/qrbloom-gallery.log 2>&1 &
	@sleep 2 && echo "gallery → http://localhost:$(PORT)/  (log: /tmp/qrbloom-gallery.log)"

gallery-gpu: ## Start the gallery on :$(PORT) picking the first free GPU.
	@nohup $(PY) gallery.py --port $(PORT) > /tmp/qrbloom-gallery.log 2>&1 &
	@sleep 2 && echo "gallery (GPU) → http://localhost:$(PORT)/"

stop-gallery: ## Kill the gallery server.
	@pkill -KILL -f "python.*gallery.py" 2>/dev/null || true
	@echo "gallery stopped"

# ── Monitoring ───────────────────────────────────────────────────────────

status: ## GPU usage + live trainings + latest val_total per version.
	@echo "── GPU ──────────────────────────────────────"
	@CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi \
	  --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
	  --format=csv,noheader
	@echo ""
	@echo "── Trainings ────────────────────────────────"
	@pgrep -af "python.*train.py" | grep -v claude || echo "  (no training running)"
	@echo ""
	@echo "── Latest epoch per version ─────────────────"
	@for V in 2 3 4 5; do \
	  L=$$(grep "val_total" runs_v$$V/train.log 2>/dev/null | tail -1); \
	  printf "  v%s: %s\n" "$$V" "$${L:-(no log)}"; \
	done

logs-v2: ## Follow v2's training log.
	@tail -f runs_v2/train.log

logs-v3: ## Follow v3's training log.
	@tail -f runs_v3/train.log

logs-v4: ## Follow v4's training log.
	@tail -f runs_v4/train.log

logs-v5: ## Follow v5's training log.
	@tail -f runs_v5/train.log

# ── Cleanup ──────────────────────────────────────────────────────────────

clean-checkpoints: ## Move checkpoints/qrbloom*.pt → backups/<ts>/.
	@mkdir -p $(BAK_DIR)
	@mv checkpoints/qrbloom*.pt $(BAK_DIR)/ 2>/dev/null || true
	@echo "checkpoints moved to $(BAK_DIR)/"

clean-runs: ## Move runs_v{N}/ → backups/<ts>/.
	@mkdir -p $(BAK_DIR)
	@mv runs_v2 runs_v3 runs_v4 runs_v5 $(BAK_DIR)/ 2>/dev/null || true
	@echo "runs_v* moved to $(BAK_DIR)/"
