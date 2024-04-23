train:
	python run.py --activation=elu --batch_size=245 --commitment_cost=0.22483239391809037 --data_dir=data/GoogleNews-T/ --init=xavier --learning_rate=0.009093320496907555 --model=TSCTM --normalisation=batch_norm --num_topic=174 --temperature=0.12615362517912884 --weight_contrast=0.888871432004498


eval:
	python utils/TU.py --data_path output/TSCTM_K50_1th_T15

