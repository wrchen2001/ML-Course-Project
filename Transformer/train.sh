export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main_informer.py \
    --model informer \
    --data ETTh1 \
    --features M \
    --target OT \
    --loss mse \
    --seq_len 96 \
    --pred_len 96 \
    --attn prob \
    --freq h \
    --use_multi_gpu \
    --device 0,1,2,3 \
    --do_predict \
