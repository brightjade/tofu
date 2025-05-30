model=phi-3.5
lr=1e-5

split=forget01
forget_loss=dpo
CUDA_VISIBLE_DEVICES=5 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}

split=forget05
forget_loss=dpo
CUDA_VISIBLE_DEVICES=5 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}

split=forget10
forget_loss=dpo
CUDA_VISIBLE_DEVICES=5 python forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr} forget_loss=${forget_loss}
