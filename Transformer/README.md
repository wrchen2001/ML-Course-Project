# Transformer预测ETT数据集
## Demo  
1. 下载ETT数据集，放在`./data`目录下
2. 安装依赖项
    ```
    pip install -r requirements.txt
    ```
3. 修改参数  
运行参数在`train.sh`中，以下为参数介绍
    > export CUDA_VISIBLE_DEVICES=0,1,2,3  #gpu id，若为单卡，可去掉这个参数  
    python -u main_informer.py \
        --model informer \  #模型  
        --data ETTh1 \  #数据集  
        --features M \ #特征  
        --target OT \ #预测目标  
        --loss mse \ #loss  
        --seq_len 96 \ #输入长度  
        --pred_len 96 \ #预测长度  
        --attn prob \  
        --freq h \  
        --use_multi_gpu \ #使用多gpu，若为单卡可去掉  
        --device 0,1,2,3 \ #gpu id  
        --do_predict \  

4. 运行  
    ```
    sh train.sh
    ```
5. 作图  
在`show_display.ipynb`中可展示模型的预测效果