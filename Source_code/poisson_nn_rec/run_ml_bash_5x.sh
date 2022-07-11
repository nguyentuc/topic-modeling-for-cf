#! /bin/bash
dataname=data_ml
data_path="../../data/$dataname"

#num_factors=50
RES=results
if [ ! -d "$RES" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $RES
fi

for l in 100
do
	#for e in 0.005
	for e in 0.005
	do
		#for drate in 0.3
		for drate in 2.0 0.8 0.7 0.6 0.3
		do
			echo "Running by ..."

			python main_tuc.py --directory $dataname --user $data_path/users_train.dat --item $data_path/items_train.dat --a 1 --b 1 --lambda_u $e --lambda_v 0.005 \
			--mult $data_path/mult.dat --theta_init $data_path/theta_xx.dat \
			--beta_init $data_path/beta_final.dat --num_factors $l --max_iter 80 --learning_rate $drate --nn 0

			#old
			#python main.py --directory $dataname --user $data_path/users_train.dat --item $data_path/items_train.dat --a 1 --b 1 --lambda_u $e --lambda_v 0.01 \
			#--mult $data_path/mult.dat --theta_init $data_path/theta_xx.dat \
			#--beta_init $data_path/beta_final.dat --num_factors $l --max_iter 100 --learning_rate $drate --nn 0

			echo "Eval $dataname rate $drate"
			python evalb.py -d $dataname -n all -r $dataname-$l-100-$drate
		done
	done
done
