export CUDA_VISIBLE_DEVICES=6

model=phi-3.5
base_path=/home/nas1_userA/minseokchoi20/research/tofu/data/MINSEOK_ft_epoch5_lr1e-05_phi-3.5_full_wd0.01/checkpoint-1250

# split=forget01_perturbed
# model_path=${base_path}/original_forget01_5
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/original_forget05_5
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget10_perturbed
# model_path=${base_path}/original_forget10_5
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget01_perturbed
# model_path=${base_path}/grad_diff_1e-05_forget01_5/checkpoint-12
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/grad_diff_1e-05_forget05_5/checkpoint-62
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget10_perturbed
# model_path=${base_path}/grad_diff_1e-05_forget10_5/checkpoint-125
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget01_perturbed
# model_path=${base_path}/npo_1e-05_forget01_5/checkpoint-12
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/npo_1e-05_forget05_5/checkpoint-62
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget10_perturbed
# model_path=${base_path}/npo_1e-05_forget10_5/checkpoint-125
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget01_perturbed
# model_path=${base_path}/idk_1e-05_forget01_5/checkpoint-12
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/idk_1e-05_forget05_5/checkpoint-62
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget10_perturbed
# model_path=${base_path}/idk_1e-05_forget10_5/checkpoint-125
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget01_perturbed
# model_path=${base_path}/dpo_1e-05_forget01_5/checkpoint-12
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/dpo_1e-05_forget05_5/checkpoint-62
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget10_perturbed
# model_path=${base_path}/dpo_1e-05_forget10_5/checkpoint-125
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget01_perturbed
# model_path=${base_path}/ot_1e-05_forget01_5/checkpoint-12
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

# split=forget05_perturbed
# model_path=${base_path}/ot_1e-05_forget05_5/checkpoint-62
# python evaluate_util.py model_family=$model split=$split model_path=$model_path

split=forget10_perturbed
model_path=${base_path}/ot_1e-05_forget10_5/checkpoint-125
python evaluate_util.py model_family=$model split=$split model_path=$model_path
