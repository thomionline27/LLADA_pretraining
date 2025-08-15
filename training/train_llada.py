# Copyright 2025 MMaDA Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_MODE'] = 'offline'
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
import torch.nn.functional as F
try:
    from lightning.pytorch.utilities import CombinedLoader
except ImportError:
    from lightning.fabric.utilities import CombinedLoader


from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from parquet import JsonlDataset
from models.modeling_llada import LLaDAModelLM
from models.configuration_llada import LLaDAConfig
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from training.prompting_utils import UniversalPrompting

from training.utils import get_config, flatten_omega_conf, AverageMeter
print("库导入完成···")

import os, wandb
os.environ["WANDB_MODE"] = "offline"
os.environ['WANDB_DIR'] = "../MMaDA/logs"
run = wandb.init(project="myproj", mode="offline")

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")
logger.info

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    
    # 使用最简单的accelerator配置
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    device = accelerator.device
    
    total_batch_size_per_gpu = config.training.batch_size
    total_batch_size = (
        config.training.batch_size
        * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize trackers
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # Set training seed
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and TOKENIZER  #
    #########################
    logger.info("Loading tokenizer and model")

    # LLaDA
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_path, padding_side="left")


    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(),
                                       ignore_id=-100, cond_dropout_prob=0, use_reserved_token=False)
    

    print('special tokens : \n', uni_prompting.sptids_dict)

    # Initialize LLaDA
    base_config = AutoConfig.from_pretrained(config.model.pretrained_model_path).to_dict()
    llada_config_dict = {k: v for k, v in config.model.llada_config.items()} if hasattr(config.model, 'llada_config') else {}
    merged_config = {**base_config, **llada_config_dict}
    llada_config = LLaDAConfig(**merged_config)
    model = LLaDAModelLM.from_pretrained(config.model.pretrained_model_path, torch_dtype=torch.bfloat16, config=llada_config)
    model.resize_token_embeddings(len(uni_prompting.text_tokenizer))
    model.config.embedding_size = model.config.vocab_size
    model = model.to(accelerator.device)

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    # No decay on bias and layernorm
    no_decay = ["bias", "layer_norm.weight", "ln_f.weight", "wte.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    dataset_config = config.dataset.params

    # LLM pure text dataset: Using JsonlDataset for .jsonl files
    dataset_lm = JsonlDataset(
        data_path=dataset_config.train_shards_path_or_url,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        num_workers=dataset_config.num_workers,
        max_length=config.dataset.preprocessing.max_seq_length
    )

    train_dataloader_lm = torch.utils.data.DataLoader(
        dataset_lm, 
        batch_size=config.training.batch_size,
        sampler=None, 
        collate_fn=dataset_lm.collate_fn,
        num_workers=dataset_config.num_workers,
    )

    estimated_samples_per_epoch = 100000
    samples_per_update_step = config.training.batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(estimated_samples_per_epoch / samples_per_update_step)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    
  

    # Combine these dataloaders into a single iterable model - exactly like train_mmada.py
    iterables = {
        "lm_flow": train_dataloader_lm,
    }

    combined_dataloader = CombinedLoader(iterables, mode="max_size_cycle")

    ##################################
    #       MODEL RESUME          #
    ##################################
    global_step = 0
    first_epoch = 0


    ##################################
    #       Prepare accelerator     #
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # mask_dtype = model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    ##################################
    logger.info("***** Running LLaDA pretraining *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Num Steps = {config.training.max_train_steps}")
    logger.info(f"  Num Epochs = {num_train_epochs}")

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        texts: Union[str, str], max_seq_len, eps=1e-3
    ):
        # create MLM mask and labels
        
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        masked_indices = noisy_batch == mask_id 
        
        return noisy_batch, labels_lm, p_mask

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:

            # for loss calculation
            batch_size_lm = len(batch["lm_flow"]["input_ids"])

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            max_seq_len = config.dataset.preprocessing.max_seq_length
            texts_lm = batch["lm_flow"]["input_ids"]
            (
                input_ids_lm,  
                labels_lm,
                p_mask_lm
            ) = prepare_inputs_and_labels_for_text(texts_lm, max_seq_len)  
            
            # For text-only: input_ids and labels are just the LM components
            input_ids = input_ids_lm.to(accelerator.device, non_blocking=True)
            labels = labels_lm.to(accelerator.device, non_blocking=True)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_lm = model.forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_lm=batch_size_lm,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    p_mask_lm=p_mask_lm,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                # print(f"GPU {accelerator.process_index} 收到的 batch_size_lm为{batch_size_lm}")
                # loss = loss_lm 
                # loss_lm_sum = loss_lm * batch_size_lm
                # length_lm_sum = batch_size_lm
                # sum_loss_lm = accelerator.gather(loss_lm_sum).sum()
                # length_lm_len = accelerator.gather(length_lm_sum).sum()
                # loss_lm = sum_loss_lm/ length_lm_len
                avg_loss_lm = accelerator.gather(loss_lm.repeat(batch_size_lm)).mean()
                loss = config.training.lm_coeff * loss_lm

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    
                    logs = {
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)


                # # Optional: Generate text samples for validation
                # if ((global_step + 1) % config.experiment.generate_every == 0 or global_step == 0) and accelerator.is_main_process:
                #     generate_text_samples(
                #         model,
                #         tokenizer,
                #         accelerator,
                #         config,
                #         global_step + 1,
                #     )

                logger.info(
                    f"Step: {global_step + 1} "
                    f"Loss_lm: {avg_loss_lm.item():0.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

        if global_step >= config.training.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()


@torch.no_grad()
def generate_text_samples(model, tokenizer, accelerator, config, global_step):
    """Generate text samples for validation during training"""
    logger.info("Generating text samples...")
    print(f"[gen] enter generate_text_samples: step={global_step}, rank={accelerator.process_index}")
    model.eval()

    # Sample prompts for text generation
    validation_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology dominates",
        "Science has always been",
        "The most important discovery in human history",
        "When we look at the stars, we see"
    ]

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    generated_texts = []
    
    for i, prompt in enumerate(validation_prompts):
        print(f"[gen] prompt[{i}] start: {prompt[:80]}{'...' if len(prompt)>80 else ''}")
        
        # Tokenize the prompt
        input_ids = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=False
        )["input_ids"].to(accelerator.device)
        
        print(f"[gen] prompt[{i}] tokenized: shape={tuple(input_ids.shape)}")
        
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            try:
                # Generate text continuation
                t0 = time.time()
                print(f"[gen] prompt[{i}] calling generate() ...")
                output_ids = accelerator.unwrap_model(model).generate(
                    input_ids=input_ids,
                    max_length=input_ids.shape[1] + 100,  # Generate 100 more tokens
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                dt = time.time() - t0
                print(f"[gen] prompt[{i}] generate() done in {dt:.2f}s, output_ids shape={tuple(output_ids.shape)}")
            except Exception as e:
                print(f"[gen] error during generation for prompt {i}: {e}")
                generated_texts.append(f"Error: {str(e)}")
                continue
        
        # Decode the generated text
        generated_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        print(f"[gen] prompt[{i}] decoded length={len(generated_text)}, content={generated_text}")
        generated_texts.append(generated_text)
    
    # Log generated texts
    for i, (prompt, generated) in enumerate(zip(validation_prompts, generated_texts)):
        wandb.log({
            f"generated_text_{i}": wandb.Html(f"<b>Prompt:</b> {prompt}<br><b>Generated:</b> {generated}")
        }, step=global_step)
    
    print(f"[gen] exit generate_text_samples: step={global_step}")
    model.train()


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # Before saving state, check if this save would set us over the checkpoints_total_limit
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # Before we save the new checkpoint, we need to have at most checkpoints_total_limit - 1 checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # Retrieve the model on all processes for deepspeed stage 3 to work then save on one process
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main() 