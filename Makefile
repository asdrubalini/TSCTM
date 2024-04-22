train:
	python run.py --data_dir data/GoogleNews-T/ --model TSCTM --num_topic 200 --learning_rate 0.002 --commitment_cost 0.1 --temperature 0.5 --weight_contrast 1.0 --normalisation batch_norm --init xavier --activation softplus

eval:
	python utils/TU.py --data_path output/TSCTM_K50_1th_T15

