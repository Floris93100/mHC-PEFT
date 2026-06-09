# train.py
""" trains model and logs all relevant info """

import os
import json
import torch
# https://huggingface.co/docs/transformers/main_classes/trainer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from models.injection import count_trainable_parameters

class DualLRTrainer(Trainer):
    """Trainer with separate learning rates for SHC and PEFT params."""
    def __init__(self, *args, shc_lr, lora_lr, **kwargs):
        """Initialize trainer with separate learning rates for parameter groups."""
        super().__init__(*args, **kwargs)
        self.shc_lr = shc_lr
        self.lora_lr = lora_lr

    def create_optimizer(self):
        """Create optimizer with separate LR groups for SHC vs PEFT params."""
        shc_params = []
        lora_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(x in name for x in ("attn_hc", "mlp_hc")):
                shc_params.append(param)
            else:
                lora_params.append(param)

        optimizer_grouped_parameters = [
            {"params": shc_params, "lr": self.shc_lr},
            {"params": lora_params, "lr": self.lora_lr},
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        print("\nOptimizer parameter groups:")
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"Group {i}:")
            print(f"  lr = {group['lr']}")
            print(f"  num params = {sum(p.numel() for p in group['params']):,}")

        return self.optimizer


def get_dataset_name(experiment_cfg):
    """ helper function to correctly set dataset name """
    if experiment_cfg.data.source_type == "tiny":
        return "tiny"
    return experiment_cfg.data.hf_dataset_name


def save_training_summary(model, metrics, train_cfg, experiment_cfg, output_dir):
    """ saves one .json summary of the training run """
    os.makedirs(output_dir, exist_ok = True)

    parameter_counts = count_trainable_parameters(model)

    summary = {
        "project_name" : experiment_cfg.project_name,
        "run_name" : experiment_cfg.run_name,
        "seed" : experiment_cfg.seed,

        "model" : experiment_cfg.model.pretrained_model_name_or_path,
        "method" : experiment_cfg.method.selected_method,
        "dataset" : get_dataset_name(experiment_cfg),
        "data_source" : experiment_cfg.data.source_type,

        "max_train_steps" : train_cfg.max_train_steps,
        "num_train_epochs" : train_cfg.num_train_epochs,
        "learning_rate" : train_cfg.learning_rate,
        "per_device_train_batch_size" : train_cfg.per_device_train_batch_size,
        "gradient_accumulation_steps" : train_cfg.gradient_accumulation_steps,

        "train_loss" : metrics.get("train_loss"),
        "train_runtime" : metrics.get("train_runtime"),
        "train_samples_per_second" : metrics.get("train_samples_per_second"),
        "train_steps_per_second" : metrics.get("train_steps_per_second"),

        "eval_loss" : metrics.get("eval_loss"),
        "validation_samples" : experiment_cfg.data.validation_samples,
        "validation_split_strategy" : experiment_cfg.data.validation_split_strategy,
        "validation_seed" : experiment_cfg.data.validation_seed,

        "test_samples" : experiment_cfg.data.test_samples,
        "split_indices_output_file" : experiment_cfg.data.split_indices_output_file,
        "split_indices_input_file" : experiment_cfg.data.split_indices_input_file,

        **parameter_counts,
    }

    with open(os.path.join(output_dir, "training_summary.json"), "w") as file:
        json.dump(summary, file, indent = 2)


def save_trainable_parameter_metadata(model, output_dir):
    """ saves small summary of trainable params """
    metadata = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            metadata.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "num_parameters": parameter.numel(),
                    "dtype": str(parameter.dtype),
                }
            )
    with open(os.path.join(output_dir, "trainable_param_metadata.json"), "w") as file:
        json.dump(metadata, file, indent = 2)


def save_trainable_parameters(model, tokenizer, output_dir):
    """ saves only params that were updated during training """
    os.makedirs(output_dir, exist_ok = True)

    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    torch.save(
        trainable_state,
        os.path.join(output_dir, "trainable_params.pt"),
    )

    with open(os.path.join(output_dir, "trainable_param_names.txt"), "w") as file:
        for name in trainable_state:
            file.write(name + "\n")

    save_trainable_parameter_metadata(model, output_dir)
    tokenizer.save_pretrained(output_dir)


def save_reload_metadata(experiment_cfg, output_dir):
    """ saves metadata needed to reload trained parameters """
    os.makedirs(output_dir, exist_ok = True)

    method_name = experiment_cfg.method.selected_method

    with open(os.path.join(output_dir, "method.txt"), "w") as file:
        file.write(method_name + "\n")

    reload_metadata = {
        "method" : method_name,
        "model" : experiment_cfg.model.pretrained_model_name_or_path,
        "torch_dtype" : experiment_cfg.model.torch_dtype,
        "device_map" : experiment_cfg.model.device_map,
        "trust_remote_code" : experiment_cfg.model.trust_remote_code,
        "shc_num_streams" : experiment_cfg.method.shc_num_streams,
        "shc_sinkhorn_iterations" : experiment_cfg.method.shc_sinkhorn_iterations,
        "shc_sinkhorn_epsilon" : experiment_cfg.method.shc_sinkhorn_epsilon,
        "shc_train_wrapped_branch" : experiment_cfg.method.shc_train_wrapped_branch,
        "shc_dropout_stream" : experiment_cfg.method.shc_dropout_stream,
        "shc_dropout_res" : experiment_cfg.method.shc_dropout_res,
        "shc_noise_std" : experiment_cfg.method.shc_noise_std,
        "shc_ablation_mapping" : experiment_cfg.method.shc_ablation_mapping,
        "shc_softmax_readout" : experiment_cfg.method.shc_softmax_readout,
        "mhc_num_fracs" : experiment_cfg.method.mhc_num_fracs,
        "peft_lora_rank" : experiment_cfg.method.peft_lora_rank,
        "peft_lora_alpha" : experiment_cfg.method.peft_lora_alpha,
        "peft_lora_dropout" : experiment_cfg.method.peft_lora_dropout,
        "peft_vera_rank" : experiment_cfg.method.peft_vera_rank,
        "peft_vera_dropout" : experiment_cfg.method.peft_vera_dropout,
        "peft_vera_projection_prng_key" : experiment_cfg.method.peft_vera_projection_prng_key,
        "peft_bias" : experiment_cfg.method.peft_bias,
        "peft_target_modules" : list(experiment_cfg.method.peft_target_modules),
        "peft_feedforward_modules": list(experiment_cfg.method.peft_feedforward_modules),  
        "shc_ablation_mapping": list(experiment_cfg.method.shc_ablation_mapping), 
        "prompt_tuning_num_virtual_tokens" : experiment_cfg.method.prompt_tuning_num_virtual_tokens,
        "prompt_tuning_init_text" : experiment_cfg.method.prompt_tuning_init_text,
        "layer_tuning_train_last_n_layers" : experiment_cfg.method.layer_tuning_train_last_n_layers,
        "layer_tuning_train_lm_head" : experiment_cfg.method.layer_tuning_train_lm_head,
        "layer_tuning_train_norm" : experiment_cfg.method.layer_tuning_train_norm,
    }

    with open(os.path.join(output_dir, "reload_metadata.json"), "w") as file:
        json.dump(reload_metadata, file, indent = 2)


def train_model(model, tokenizer, train_dataset, cfg, experiment_cfg, eval_dataset = None):
    """ runs finetuning """
    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,                    # causal LM, not masked LM
                                        # for A100 GPU, multiples of 64 are preferred
        # https://huggingface.co/docs/transformers/v4.37.2/perf_train_gpu_onehttps://huggingface.co/docs/transformers/v4.37.2/perf_train_gpu_one
        pad_to_multiple_of = experiment_cfg.data.pad_to_multiple_of
    )
    eval_strategy = cfg.eval_strategy if eval_dataset is not None else "no"
    eval_steps = cfg.eval_steps if eval_dataset is not None else None

    args = TrainingArguments(
        output_dir = cfg.checkpoint_output_dir,
        num_train_epochs = cfg.num_train_epochs,
        max_steps = cfg.max_train_steps,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        logging_steps = cfg.logging_steps,
        logging_first_step = cfg.logging_first_step,
        save_strategy = cfg.save_strategy,
        save_total_limit = cfg.save_total_limit,
        report_to = cfg.report_to,
        logging_dir = cfg.logging_dir,
        run_name = cfg.tensorboard_run_name,
        remove_unused_columns = False,
        logging_strategy = "steps",
        disable_tqdm = cfg.disable_tqdm,
        seed = cfg.seed,                # two seeds b/c Hugging Face separates
        data_seed = cfg.seed,           # general- from data--order-randomness
        optim = cfg.optimizer,
        lr_scheduler_type = cfg.lr_scheduler_type,
                                        # how many % of training steps are warmup steps;
                                        # if training unstable or early loss spikes -> increase
        warmup_ratio = cfg.warmup_ratio,
        max_grad_norm = cfg.max_grad_norm,
        bf16 = cfg.bf16,
        fp16 = cfg.fp16,
        eval_strategy = eval_strategy,
        eval_steps = eval_steps,
        per_device_eval_batch_size = experiment_cfg.eval.per_device_eval_batch_size,
        save_steps = cfg.save_steps,
    )

    if cfg.use_dual_lr: 
        print("\n=== DualLRTrainer param groups ===")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_shc = any(x in name for x in ("attn_hc", "mlp_hc"))
            print(f"  {'SHC ' if is_shc else 'LoRA'} | lr={'shc_lr' if is_shc else 'lora_lr'} | {name}")
        print(f"  shc_lr  = {cfg.shc_learning_rate}")
        print(f"  lora_lr = {cfg.lora_learning_rate}")
        print("==================================\n")
        trainer = DualLRTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=tokenizer,
            shc_lr=cfg.shc_learning_rate,
            lora_lr=cfg.lora_learning_rate,
            )
    else: 
        trainer = Trainer(
            model = model,
            args = args,
            train_dataset = train_dataset,
            eval_dataset = eval_dataset,
            data_collator = collator,
            processing_class = tokenizer,
        )

    train_output = trainer.train(       # possibly resume from checkpoint
        resume_from_checkpoint = cfg.resume_from_checkpoint,
    )
    metrics = train_output.metrics      # also save validation loss (just for test runs)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    save_trainable_parameters(
        model,
        tokenizer,
        cfg.final_trainable_params_dir,
    )
    save_reload_metadata(
        experiment_cfg,
        cfg.final_trainable_params_dir,

    )
    save_training_summary(
        model,
        metrics,
        cfg,
        experiment_cfg,
        cfg.final_trainable_params_dir,
    )
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return trainer, metrics