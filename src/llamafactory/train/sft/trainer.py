# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
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

import json
import os
import time
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def _save_checkpoint(self, model, trial):
        r"""Override to measure checkpoint saving time."""
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        logger.info_rank0(f"💾 Starting checkpoint save at step {self.state.global_step}...")
        start_time = time.time()
        
        # Call the parent class's _save_checkpoint method
        result = super()._save_checkpoint(model, trial)
        
        elapsed_time = time.time() - start_time
        logger.info_rank0(
            f"✅ Checkpoint saved to {output_dir} in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        )
        
        return result
    
    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        r"""
        Override to save LoRA adapter with DeepSpeed ZeRO-3 compatibility.
        
        Uses direct torch.distributed.all_gather on ds_tensor to avoid
        GatheredParameters INFLIGHT issues with MoE models after evaluation.
        """
        from peft import PeftModel
        from collections import OrderedDict
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if isinstance(unwrapped_model, PeftModel) and self.is_deepspeed_enabled:
            logger.info_rank0(f"💡 Saving PEFT adapter with DeepSpeed ZeRO-3 (direct all_gather)...")
            
            # Collect LoRA params
            lora_params = [(name, param) for name, param in unwrapped_model.named_parameters()
                          if param.requires_grad and 'lora' in name.lower()]
            logger.info_rank0(f"📊 Found {len(lora_params)} LoRA params")
            
            # Sync all ranks before gathering
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            torch.cuda.synchronize()
            
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            
            state_dict = OrderedDict()
            
            for name, param in lora_params:
                if hasattr(param, 'ds_tensor'):
                    # ZeRO-3: param is partitioned, gather from all ranks
                    local_tensor = param.ds_tensor.to(param.device)
                    
                    # Create list of tensors to gather into
                    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                    torch.distributed.all_gather(gathered_tensors, local_tensor)
                    
                    # Concatenate all shards to get full tensor
                    full_tensor = torch.cat(gathered_tensors, dim=0)
                    
                    # Reshape to original shape
                    full_tensor = full_tensor[:param.ds_numel].view(param.ds_shape)
                    
                    if self.args.should_save:
                        state_dict[name] = full_tensor.detach().cpu().clone()
                else:
                    # Not partitioned
                    if self.args.should_save:
                        state_dict[name] = param.detach().cpu().clone()
            
            logger.info_rank0(f"📊 Gathered {len(state_dict)} params successfully!")
            
            # Verify values - check for all-zeros
            if self.args.should_save:
                lora_a_zeros = sum(1 for n, t in state_dict.items() if 'lora_A' in n and (t == 0).all())
                lora_b_zeros = sum(1 for n, t in state_dict.items() if 'lora_B' in n and (t == 0).all())
                lora_a_total = sum(1 for n in state_dict if 'lora_A' in n)
                lora_b_total = sum(1 for n in state_dict if 'lora_B' in n)
                logger.info_rank0(f"   lora_A: {lora_a_total} params, {lora_a_zeros} all-zeros")
                logger.info_rank0(f"   lora_B: {lora_b_total} params, {lora_b_zeros} all-zeros")
                
                if lora_b_zeros == lora_b_total and lora_b_total > 0:
                    logger.info_rank0(f"⚠️ WARNING: All lora_B weights are zero! Check if learning_rate > 0")
                
                unwrapped_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
                logger.info_rank0(f"✅ Adapter saved!")
            
            # Sync all ranks
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            if self.processing_class is not None and self.args.should_save:
                self.processing_class.save_pretrained(output_dir)
        
        elif isinstance(unwrapped_model, PeftModel):
            # Non-DeepSpeed PEFT save
            unwrapped_model.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
            )
            if self.processing_class is not None and self.args.should_save:
                self.processing_class.save_pretrained(output_dir)
            logger.info_rank0(f"✅ Adapter saved!")
        
        else:
            # Non-PEFT models
            super().save_model(output_dir, _internal_call)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
