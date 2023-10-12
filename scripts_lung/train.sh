cd ..
python train_SSMMAM.py --gpu 3 --out exp/lung  --scale  --lr 1e-4 --n-labeled 40 --consistency 1.0 --consistency_rampup 600 --epochs 20   --batch-size 16  \
--num-class 2  --val-iteration 10
