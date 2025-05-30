model=llama-3.1-8b
lr=1e-5

split=forget05
forget_loss=ot
CUDA_VISIBLE_DEVICES=7 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}

# split=forget01
# forget_loss=ot
# CUDA_VISIBLE_DEVICES=6,7 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}

# split=forget10
# forget_loss=ot
# CUDA_VISIBLE_DEVICES=6,7 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}
