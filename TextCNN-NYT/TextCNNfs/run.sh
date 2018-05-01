if [ ! -d features_0 ]; then
    echo "features_0 directory doesn't exist"
    exit 1
fi

if [ -z "$1+x" ]; then
    1=0
fi

echo "Run [t/all]:"
read ans
if [ "$ans" == "t" ]; then
    run_type=n
elif [ "$ans" == "all" ]; then
    run_type=y
else
    exit 1
fi

if [ -f RUN_LOG ]; then
	echo "RUN_LOG exists, delete? [y/other]"
	read ans
	if [ "$ans" == "y" ]; then
		rm RUN_LOG
	else
		exit 1
	fi
fi
mkdir tmp
cp -r features_0/* tmp
echo "1">tmp/time_list
for ((i=$1;i<12;i++))
do
	echo "`date -R` RUN TIME $i" | tee -a RUN_LOG
	rm TRAIN_SUCCEED
	rm EVAL_SUCCEED

	rm -rf tmp/train
	rm -rf tmp/eval
	mkdir tmp/train
	mkdir tmp/eval

	echo "`date -R` Start train..." | tee -a RUN_LOG
	echo "y
$i
$run_type" | python3 TextCNNfs_train.py &
	TRAIN_PID=$!
	echo "TRAIN_PID:$TRAIN_PID"
	
	echo "`date -R` Start eval..." | tee -a RUN_LOG
	echo "y
$i" | python3 TextCNNfs_eval.py &
	EVAL_PID=$!
	echo "EVAL_PID:$EVAL_PID"
	
	echo "`date -R` waiting train & eval: $TRAIN_PID, $EVAL_PID" | tee -a RUN_LOG
	wait -n $TRAIN_PID $EVAL_PID
	
	if [ ! -f TRAIN_SUCCEED ]; then

		if ps -p $TRAIN_PID > /dev/null; then
			echo "`date -R` EVAL exits improperly when train running, break..." | tee -a RUN_LOG
			kill $TRAIN_PID
			tmux detach
			break
		fi

		echo "`date -R` TRAIN exits improperly, break..." | tee -a RUN_LOG
		kill $EVAL_PID
		tmux detach
		break
	fi

	echo "`date -R` waiting eval: $EVAL_PID" | tee -a RUN_LOG
	wait $EVAL_PID
	if [ ! -f EVAL_SUCCEED ]; then
		echo "`date -R` EVAL exits improperly after train finished" | tee -a RUN_LOG
	fi
	
	rm -r tmp/train_$i
	rm -r tmp/eval_$i
	cp -r tmp/train tmp/train_$i
	cp -r tmp/eval tmp/eval_$i

	
	echo "Start feature..."
	echo "0 1 2 3 4 5 6 7 8 9 10 11" | python3 TextCNNfs_train_feature.py

	if [[ $? != 0 ]]; then
        echo "feature exits improperly, restarting..." | tee -a RUN_LOG
        tmux detach;
        let "i-=1";
        break
    fi
	
	rm -r tmp/train
	rm -r tmp/eval

done