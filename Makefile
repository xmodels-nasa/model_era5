.PHONY: run ft raw-chips baseline-train

FT_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings.py
RAW_CHIPS_SCRIPT := baseline_model/get_raw_chips_from_all_feather_files.py
BASELINE_TRAIN_SCRIPT := baseline_model/train_multilabel_from_raw_chips.py
LOG_DIR := logs

run: ft

ft:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_$${timestamp}.log"; \
	nohup python3 "$(FT_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"

raw-chips:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_raw_chips_$${timestamp}.log"; \
	nohup .venv/bin/python "$(RAW_CHIPS_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started raw-chip extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_raw_chips_$${timestamp}.log"; \
	nohup .venv/bin/python "$(BASELINE_TRAIN_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"
