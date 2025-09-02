python .\scripts\prediction.py --config-path .\configs\baseline\predict_without_finetuning_2_5_cm.toml
python .\scripts\prediction.py --config-path .\configs\baseline\predict_without_finetuning_5_cm.toml
python .\scripts\prediction.py --config-path .\configs\baseline\predict_without_finetuning_7_5_cm.toml
python .\scripts\prediction.py --config-path .\configs\baseline\predict_without_finetuning_10_cm.toml

python .\scripts\evaluate.py --config-path .\configs\baseline\evaluate_without_finetuning_2_5_cm.toml
python .\scripts\evaluate.py --config-path .\configs\baseline\evaluate_without_finetuning_5_cm.toml
python .\scripts\evaluate.py --config-path .\configs\baseline\evaluate_without_finetuning_7_5_cm.toml
python .\scripts\evaluate.py --config-path .\configs\baseline\evaluate_without_finetuning_10_cm.toml