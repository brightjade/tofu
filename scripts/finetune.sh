master_port=18765
split=full
model=llama-3.1-8b
lr=1e-5
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=1 gradient_accumulation_steps=8 model_family=${model} lr=${lr}
CUDA_VISIBLE_DEVICES=2 python finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}