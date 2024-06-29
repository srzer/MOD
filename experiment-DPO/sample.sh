for ((i=0; i<=10; i++))
do
    a=$(echo "scale=1; $i" | bc)
    b=$(echo "scale=1; 1-$a" | bc)
    CUDA_VISIBLE_DEVICES=0 bash scripts/eval/eval.sh 0 fusion reverse_kl $a $b
    CUDA_VISIBLE_DEVICES=1 bash scripts/eval/eval.sh 1 reward_soup reverse_kl $a $b
done