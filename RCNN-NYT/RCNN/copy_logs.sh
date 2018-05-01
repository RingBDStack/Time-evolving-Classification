rsync -avm --include='log_eval' --include='best_eval' --include='log_train' -f 'hide,! */' . logs
mv logs/tmp _logs
rm -r logs
mv _logs logs