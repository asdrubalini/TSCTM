train:
	python run.py --activation=softplus --batch_size=256 --commitment_cost=0.9 --data_dir=data/pants/ --init=xavier --learning_rate=0.007 --model=TSCTM --normalisation=batch_norm --num_topic=12 --temperature=0.12 --weight_contrast=0.75


eval:
	python utils/TU.py --data_path output/TSCTM_K50_1th_T15

