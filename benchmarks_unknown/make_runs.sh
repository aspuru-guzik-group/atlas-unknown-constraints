#! /bin/bash

for feas_strategy in 'naive' 'fwa' 'fca' 'fia'; do
	

	if [[ "$feas_strategy" == "naive" ]]; then
		naive_feas="True"
		feas_value="1"  # not used
		feas_approach="fwa"  # not used

		folder="${feas_strategy}-0"
		if [ ! -d $folder ]; then
			echo "Creating  ${folder}"
			mkdir $folder
		fi

		script="$folder/run.py"
		sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
		sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
		sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
		cp submit.sh "$folder/"

	elif [[ "$feas_strategy" == "fwa" ]]; then
		naive_feas="False"
		feas_value="1"  # not used
		feas_approach="fwa"

		folder="${feas_strategy}-0"
		if [ ! -d $folder ]; then
			echo "Creating  ${folder}"
			mkdir $folder
		fi

		script="$folder/run.py"
		sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
		sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
		sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
		cp submit.sh "$folder/"

	elif [[ "$feas_strategy" == "fca" ]]; then
		naive_feas="False"
		feas_approach="fca"

		for feas_value in "0.2" "0.5" "0.8"; do

			folder="${feas_strategy}-${feas_value}"
			if [ ! -d $folder ]; then
				echo "Creating  ${folder}"
				mkdir $folder
			fi			

			script="$folder/run.py"
			sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
			sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
			sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
			cp submit.sh "$folder/"
		done

	elif [[ "$feas_strategy" == "fia" ]]; then
		naive_feas="False"
		feas_approach="fia"

		for feas_value in "0.5" "1" "2" "1000"; do

			folder="${feas_strategy}-${feas_value}"
			if [ ! -d $folder ]; then
				echo "Creating  ${folder}"
				mkdir $folder
			fi			

			script="$folder/run.py"
			sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
			sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
			sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
			cp submit.sh "$folder/"
		done
	fi
done
