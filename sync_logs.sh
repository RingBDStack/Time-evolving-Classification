source activate tf-test
rm -r logs
mkdir logs
exps=()
while IFS='' read -r line || [[ -n "$line" ]]; do
	exps+=($line)
done < exp_paths_list.txt
total=${#exps[*]}
for ((k=0;k<total;k++))
do
	exp=${exps[$k]}
	echo "Read paths from $exp"
	paths=()
	while IFS='' read -r line || [[ -n "$line" ]]; do
		paths+=($line)
	done < $exp
	echo "Read psw"
	psw=()
	while IFS='' read -r line || [[ -n "$line" ]]; do
		psw+=($line)
	done < psw.txt
	n=${#paths[*]}
	rm -r logs/$exp
	mkdir logs/$exp
	for ((i=0;i<n;i++))
	do
		p=${paths[$i]}
		name=${p##*/}
		echo "Copying from $p to $name"
		if [ -d "logs/${exp}/$name" ]; then
			echo "duplicate name: $name"
			mv logs/${exp}/$name logs/${exp}/${name}_
		fi
	
		sshpass -p ${psw[$i]} rsync -avmq --include='log_eval' --include='best_eval' --include='_option.py' --include='log_train' --include="RUN_LOG" --include="*.log" --exclude "log" -f 'hide,! */' $p logs/$exp
		cp -r logs/${exp}/${name}/tmp/* logs/${exp}/${name}/
		rm -r logs/${exp}/${name}/tmp
		if [ -d "logs/${exp}/${name}_" ]; then
			mv logs/${exp}/$name logs/${exp}/${name}_$i
			mv logs/${exp}/${name}_ logs/${exp}/$name
		fi
	done
done
echo "Gather results..."
python gather_result.py
