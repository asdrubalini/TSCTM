train:
	python run.py --data_dir data/GoogleNews-T/ --model TSCTM --num_topic 200

eval:
	python utils/TU.py --data_path output/TSCTM_K50_1th_T15

