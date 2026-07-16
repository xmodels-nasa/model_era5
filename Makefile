.PHONY: run ft ft-transformer ft-transformer-no-lat-lon ft-transformer-no-time ft-transformer-local-solar-time ft-forecast ft-3x3 embedding embedding-forecast raw-chips raw-chips-forecast baseline-train baseline-train-forecast baseline-train-aurora baseline-train-aurora-forecast

FT_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings.py
FT_TRANSFORMER_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings_transformer.py
FT_TRANSFORMER_NO_LAT_LON_SCRIPT := fine_tune_model_no_lat_lon_feature/train_multilabel_from_feather_embeddings_transformer_no_lat_lon.py
FT_TRANSFORMER_NO_LAT_LON_OUTPUT_DIR := results-v3/model_outputs_transformer_no_lat_lon
FT_TRANSFORMER_NO_TIME_SCRIPT := fine_tune_model_no_lat_lon_feature/train_multilabel_from_feather_embeddings_transformer_no_time.py
FT_TRANSFORMER_NO_TIME_OUTPUT_DIR := results-v3/model_outputs_transformer_no_time
FT_TRANSFORMER_LOCAL_SOLAR_TIME_SCRIPT := fine_tune_model_no_lat_lon_feature/train_multilabel_from_feather_embeddings_transformer_local_solar_time.py
FT_TRANSFORMER_LOCAL_SOLAR_TIME_OUTPUT_DIR := results-v3/model_outputs_transformer_local_solar_time
FT_FORECAST_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings_forecast.py
FT_3X3_SCRIPT := fine_tuned_model/train_multilabel_from_feather_embeddings_3x3.py
FT_3X3_ARGS := --sample-ratio 0.5
EMBEDDING_SCRIPT := fine_tuned_model/get_embedings_from_all_feather_files_3_by_3_grids.py
EMBEDDING_FORECAST_SCRIPT := fine_tuned_model/get_embedings_from_all_feather_files_forecast.py
RAW_CHIPS_SCRIPT := baseline_model/get_raw_chips_from_all_feather_files.py
RAW_CHIPS_FORECAST_SCRIPT := baseline_model/get_raw_chips_from_all_feather_files_forecast.py
BASELINE_TRAIN_SCRIPT := baseline_model/train_multilabel_from_raw_chips.py
BASELINE_TRAIN_FORECAST_SCRIPT := baseline_model/train_multilabel_from_raw_chips_forecast.py
BASELINE_AURORA_TRAIN_SCRIPT := baseline_model/train_multilable_from_rawchips_aurora_architecturer.py
BASELINE_AURORA_TRAIN_FORECAST_SCRIPT := baseline_model/train_multilable_from_rawchips_aurora_architecturer_forecast.py
LOG_DIR := logs
TRAIN_FILES := 1000
LOCAL_SOLAR_TIME_TRAIN_FILES := 3000
LOCAL_SOLAR_TIME_SAMPLE_RATIO := 0.1

run: ft

ft:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(FT_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"

ft-transformer:
	@mkdir -p "$(LOG_DIR)" ".matplotlib"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_transformer_$${timestamp}.log"; \
	MPLCONFIGDIR="$(CURDIR)/.matplotlib" PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(FT_TRANSFORMER_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started transformer embedding fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"

ft-transformer-no-lat-lon:
	@mkdir -p "$(LOG_DIR)" ".matplotlib" "$(FT_TRANSFORMER_NO_LAT_LON_OUTPUT_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_transformer_no_lat_lon_$${timestamp}.log"; \
	MPLCONFIGDIR="$(CURDIR)/.matplotlib" PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(FT_TRANSFORMER_NO_LAT_LON_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started no-lat/lon transformer embedding fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"; \
	echo "Output directory: $(FT_TRANSFORMER_NO_LAT_LON_OUTPUT_DIR)"

ft-transformer-no-time:
	@mkdir -p "$(LOG_DIR)" ".matplotlib" "$(FT_TRANSFORMER_NO_TIME_OUTPUT_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_transformer_no_time_$${timestamp}.log"; \
	MPLCONFIGDIR="$(CURDIR)/.matplotlib" PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(FT_TRANSFORMER_NO_TIME_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started no-time transformer embedding fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"; \
	echo "Output directory: $(FT_TRANSFORMER_NO_TIME_OUTPUT_DIR)"

ft-transformer-local-solar-time:
	@mkdir -p "$(LOG_DIR)" ".matplotlib" "$(FT_TRANSFORMER_LOCAL_SOLAR_TIME_OUTPUT_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_transformer_local_solar_time_$${timestamp}.log"; \
	MPLCONFIGDIR="$(CURDIR)/.matplotlib" PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(FT_TRANSFORMER_LOCAL_SOLAR_TIME_SCRIPT)" --train-files "$(LOCAL_SOLAR_TIME_TRAIN_FILES)" --sample-ratio "$(LOCAL_SOLAR_TIME_SAMPLE_RATIO)" > "$$log_file" 2>&1 & \
	echo "Started local-solar-time transformer embedding fine-tuning in background. PID=$$!"; \
	echo "Log file: $$log_file"; \
	echo "Train files: $(LOCAL_SOLAR_TIME_TRAIN_FILES)"; \
	echo "Sample ratio: $(LOCAL_SOLAR_TIME_SAMPLE_RATIO)"; \
	echo "Output directory: $(FT_TRANSFORMER_LOCAL_SOLAR_TIME_OUTPUT_DIR)"

ft-forecast:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_feather_embeddings_forecast_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(FT_FORECAST_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started forecast fine-tuning in background. PID=$$!"; \
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

embedding-forecast:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_embeddings_forecast_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup python3 -u "$(EMBEDDING_FORECAST_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started forecast embedding extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

raw-chips:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_raw_chips_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(RAW_CHIPS_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started raw-chip extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

raw-chips-forecast:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/get_raw_chips_forecast_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(RAW_CHIPS_FORECAST_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started forecast raw-chip extraction in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_raw_chips_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(BASELINE_TRAIN_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train-forecast:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilabel_from_raw_chips_forecast_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(BASELINE_TRAIN_FORECAST_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started forecast baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train-aurora:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilable_from_rawchips_aurora_architecturer_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(BASELINE_AURORA_TRAIN_SCRIPT)" --train-files "$(TRAIN_FILES)" > "$$log_file" 2>&1 & \
	echo "Started Aurora-style baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"

baseline-train-aurora-forecast:
	@mkdir -p "$(LOG_DIR)"
	@timestamp="$$(date +"%Y%m%d_%H%M%S")"; \
	log_file="$(LOG_DIR)/train_multilable_from_rawchips_aurora_architecturer_forecast_$${timestamp}.log"; \
	PYTHONUNBUFFERED=1 nohup .venv/bin/python -u "$(BASELINE_AURORA_TRAIN_FORECAST_SCRIPT)" > "$$log_file" 2>&1 & \
	echo "Started forecast Aurora-style baseline raw-chip training in background. PID=$$!"; \
	echo "Log file: $$log_file"
