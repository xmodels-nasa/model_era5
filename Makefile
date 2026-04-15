.PHONY: run ft

FT_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings.py
LOG_DIR := logs

run: ft

ft:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_$${timestamp}.log"; \
	nohup python3 "$(FT_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"
