#! /bin/bash

for gryffin_strategy in "naive" "static" "dynamic"; do

	# Desciptor keys
	if [[ "$gryffin_strategy" == "naive" ]]; then
		decsriptors="False"
		dynamic="False"
	elif [[ "$gryffin_strategy" == "static" ]]; then
		decsriptors="True"
		dynamic="False"
	elif [[ "$gryffin_strategy" == "dynamic" ]]; then
		decsriptors="True"
		dynamic="True"
	fi


for feas_strategy in 'naive' 'fwa' 'fca' 'fia'; do
	

	if [[ "$feas_strategy" == "naive" ]]; then
		naive_feas="True"
		feas_value="1"  # not used
		feas_approach="fwa"  # not used

		folder="${gryffin_strategy}_gryffin-${feas_strategy}-0"
		if [ ! -d $folder ]; then
			echo "Creating  ${folder}"
			mkdir $folder
		fi

		script="$folder/run.py"

		sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
		sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
		sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
		sed -i.bak "s/_DESCRIPTORS_/${decsriptors}/g" $script && rm $script.bak
		sed -i.bak "s/_DYNAMIC_/${dynamic}/g" $script && rm $script.bak
		cp submit.sh "$folder/"

	elif [[ "$feas_strategy" == "fwa" ]]; then
		naive_feas="False"
		feas_value="1"  # not used
		feas_approach="fwa"

		folder="${gryffin_strategy}_gryffin-${feas_strategy}-0"
		if [ ! -d $folder ]; then
			echo "Creating  ${folder}"
			mkdir $folder
		fi

		script="$folder/run.py"
		sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
		sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
		sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
		sed -i.bak "s/_DESCRIPTORS_/${decsriptors}/g" $script && rm $script.bak
		sed -i.bak "s/_DYNAMIC_/${dynamic}/g" $script && rm $script.bak
		cp submit.sh "$folder/"

	elif [[ "$feas_strategy" == "fca" ]]; then
		naive_feas="False"
		feas_approach="fca"

		for feas_value in "0.2" "0.5" "0.8"; do

			folder="${gryffin_strategy}_gryffin-${feas_strategy}-${feas_value}"
			if [ ! -d $folder ]; then
				echo "Creating  ${folder}"
				mkdir $folder
			fi			

			script="$folder/run.py"
			sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
			sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
			sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
			sed -i.bak "s/_DESCRIPTORS_/${decsriptors}/g" $script && rm $script.bak
			sed -i.bak "s/_DYNAMIC_/${dynamic}/g" $script && rm $script.bak
			cp submit.sh "$folder/"
		done

	elif [[ "$feas_strategy" == "fia" ]]; then
		naive_feas="False"
		feas_approach="fia"

		for feas_value in "0.5" "1" "2" "1000"; do

			folder="${gryffin_strategy}_gryffin-${feas_strategy}-${feas_value}"
			if [ ! -d $folder ]; then
				echo "Creating  ${folder}"
				mkdir $folder
			fi			

			script="$folder/run.py"
			sed "s/_NAIVE_/${naive_feas}/g" "run_template.py" > $script
			sed -i.bak "s/_FEAS_/${feas_value}/g" $script && rm $script.bak
			sed -i.bak "s/_APPROACH_/'${feas_approach}'/g" $script && rm $script.bak
			sed -i.bak "s/_DESCRIPTORS_/${decsriptors}/g" $script && rm $script.bak
			sed -i.bak "s/_DYNAMIC_/${dynamic}/g" $script && rm $script.bak
			cp submit.sh "$folder/"
		done
	fi
done

done


