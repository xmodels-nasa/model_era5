.PHONY: run ft ft-3x3 embedding raw-chips baseline-train

FT_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings.py
FT_3X3_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings_3x3.py
FT_3X3_ARGS := --sample-ratio 0.5
EMBEDDING_SCRIPT := fine_tuned_model/get_embedings_from_all_feather_files_3_by_3_grids.py
RAW_CHIPS_SCRIPT := baseline_model/get_raw_chips_from_all_feather_files.py
BASELINE_TRAIN_SCRIPT := baseline_model/train_multilabel_from_raw_chips.py
LOG_DIR := logs

run: ft

ft:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(FT_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"

ft-3x3:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_3x3_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(FT_3X3_SCRIPT)" $(FT_3X3_ARGS) > "$$log_file" 2>&1 & \
	echo "Started 3x3 embedding fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"

embedding:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_embeddings_3x3_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(EMBEDDING_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started 3x3 embedding extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

raw-chips:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_raw_chips_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(RAW_CHIPS_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started raw-chip extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_raw_chips_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(BASELINE_TRAIN_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"
