#! /bin/bash
dataname=data_mv
data_path="../../data/$dataname"

num_factors=50
RES=results
if [ ! -d "$RES" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $RES
fi

#root_path=../data/arxiv/cv
for l in 50 100 150 200
do
	for e in 0.005
	do
		for drate in 0.3
		do
			echo "Run for l= $l e= $e drate= $drate"
			python main.py --directory $dataname --user $data_path/users_train.dat --item $data_path/items_train.dat --a 1 --b 1 --lambda_u $e --lambda_v 0.01 \
			--mult $data_path/mult.dat --theta_init $data_path/theta_2.dat \
			--beta_init $data_path/beta_final.dat --num_factors 100 --num_hidden $l  --max_iter 100 --learning_rate $drate
			#echo "Eval $dataname rate $drate"
			#python evalb.py -d $dataname -n all -r $dataname-$drate
			#echo "Factor: $e rate: $drate at loop: $l"
		done
	done
done
