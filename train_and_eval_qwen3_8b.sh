#!/bin/bash

set -e  # Exit on error

# Wandb configuration
WANDB_API_KEY="dc07c5d951fc5c844b15752232fde38909adec05"
WANDB_RUN_ID="filtered_520"

# Redirect all output to log file (both stdout and stderr)
LOG_FILE="/data/workspace/kimberly/LLaMA-Factory/${WANDB_RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Starting Qwen3-8B Training and Evaluation"
echo "=========================================="
echo "📝 All output will be saved to: ${LOG_FILE}"
echo "🕐 Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================
# Step 1: Training with LLaMA-Factory
# ============================================
echo ""
echo ">>> Step 1: Training Qwen3-8B with LoRA"
echo ""

cd /data/workspace/kimberly/LLaMA-Factory && source venv/bin/activate

export WANDB_API_KEY=$WANDB_API_KEY
wandb login
export WANDB_RUN_ID=$WANDB_RUN_ID

# Create temporary config with dynamic output_dir
TEMP_CONFIG="/tmp/qwen3_8b_${WANDB_RUN_ID}.yaml"
echo "Creating temporary config with output_dir: saves/${WANDB_RUN_ID}/lora/sft"

# Replace qwen3-8b with WANDB_RUN_ID in output_dir
sed "s|output_dir: saves/qwen3-8b/lora/sft|output_dir: saves/${WANDB_RUN_ID}/lora/sft|g" \
    examples/train_lora/qwen3_8b.yaml > ${TEMP_CONFIG}

# Start training
echo "Starting training with config: ${TEMP_CONFIG}"
llamafactory-cli train ${TEMP_CONFIG}

echo ""
echo "✅ Training completed!"
echo ""

# Clean up temporary config
rm -f ${TEMP_CONFIG}

# ============================================
# Step 2: Evaluation with lm-evaluation-harness
# ============================================
echo ""
echo ">>> Step 2: Evaluating Qwen3-8B on Korean benchmarks"
echo ""

deactivate && cd /data/workspace/kimberly/lm-evaluation-harness && source venv/bin/activate

# Find all checkpoints (using WANDB_RUN_ID for dynamic path)
CHECKPOINT_DIR="/data/workspace/kimberly/LLaMA-Factory/saves/${WANDB_RUN_ID}/lora/sft"
CHECKPOINTS=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoints found in ${CHECKPOINT_DIR}"
    exit 1
fi

echo "📁 Found checkpoints:"
echo "$CHECKPOINTS" | while read checkpoint; do
    echo "  - $(basename $checkpoint)"
done
echo ""

# Evaluate each checkpoint
for CHECKPOINT_PATH in $CHECKPOINTS; do
    CHECKPOINT_NAME=$(basename $CHECKPOINT_PATH)
    echo ""
    echo "=========================================="
    echo "Evaluating ${CHECKPOINT_NAME}"
    echo "=========================================="
    echo ""

    # Evaluation 1: KMMLU
    echo ""
    echo ">>> Running KMMLU evaluation for ${CHECKPOINT_NAME}..."
    echo ""

    accelerate launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        --num_machines 1 \
        --dynamo_backend no \
        -m lm_eval \
        --model hf \
        --model_args pretrained=/data/cache_dir/models/Qwen/Qwen3-8B,tokenizer=/data/cache_dir/models/Qwen/Qwen3-8B,dtype=bfloat16,peft=${CHECKPOINT_PATH},trust_remote_code=True \
        --tasks kmmlu \
        --batch_size 256 \
        --output_path ./results/kmmlu/${WANDB_RUN_ID}-${CHECKPOINT_NAME} \
        --num_fewshot 0

    echo "✅ KMMLU evaluation completed for ${CHECKPOINT_NAME}!"

    # Evaluation 2: KoBEST
    echo ""
    echo ">>> Running KoBEST evaluation for ${CHECKPOINT_NAME}..."
    echo ""

    accelerate launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        --num_machines 1 \
        --dynamo_backend no \
        -m lm_eval \
        --model hf \
        --model_args pretrained=/data/cache_dir/models/Qwen/Qwen3-8B,tokenizer=/data/cache_dir/models/Qwen/Qwen3-8B,dtype=bfloat16,peft=${CHECKPOINT_PATH},trust_remote_code=True \
        --tasks kobest \
        --batch_size 512 \
        --output_path ./results/kobest/${WANDB_RUN_ID}-${CHECKPOINT_NAME} \
        --num_fewshot 0

    echo "✅ KoBEST evaluation completed for ${CHECKPOINT_NAME}!"

    # Evaluation 3: HAE-RAE
    echo ""
    echo ">>> Running HAE-RAE evaluation for ${CHECKPOINT_NAME}..."
    echo ""

    accelerate launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        --num_machines 1 \
        --dynamo_backend no \
        -m lm_eval \
        --model hf \
        --model_args pretrained=/data/cache_dir/models/Qwen/Qwen3-8B,tokenizer=/data/cache_dir/models/Qwen/Qwen3-8B,dtype=bfloat16,peft=${CHECKPOINT_PATH},trust_remote_code=True \
        --tasks haerae \
        --batch_size 512 \
        --output_path ./results/haerae/${WANDB_RUN_ID}-${CHECKPOINT_NAME} \
        --num_fewshot 0

    echo "✅ HAE-RAE evaluation completed for ${CHECKPOINT_NAME}!"

    # Evaluation 4: Interview Eval
    echo ""
    echo ">>> Running Interview Eval for ${CHECKPOINT_NAME}..."
    echo ""

    accelerate launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        --num_machines 1 \
        --dynamo_backend no \
        -m lm_eval \
        --model hf \
        --model_args "pretrained=/data/cache_dir/models/Qwen/Qwen3-8B,tokenizer=/data/cache_dir/models/Qwen/Qwen3-8B,dtype=bfloat16,peft=${CHECKPOINT_PATH},trust_remote_code=True,think_end_token=</think>" \
        --tasks interview_eval \
        --batch_size 16 \
        --output_path ./results/interview_eval/${WANDB_RUN_ID}-${CHECKPOINT_NAME} \
        --num_fewshot 0 \
        --log_samples \
        --apply_chat_template

    echo "✅ Interview Eval completed for ${CHECKPOINT_NAME}!"
    echo ""
done

echo ""
echo "=========================================="
echo "All checkpoint evaluations completed!"
echo "=========================================="

# ============================================
# Completion
# ============================================
echo ""
echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="
echo "🕐 End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 Results:"
echo "  - Full execution log: ${LOG_FILE}"
echo ""
echo "📁 Evaluation results by checkpoint:"
echo "$CHECKPOINTS" | while read checkpoint; do
    checkpoint_name=$(basename $checkpoint)
    echo "  📌 ${checkpoint_name}:"
    echo "     - KMMLU: ./results/kmmlu/${WANDB_RUN_ID}-${checkpoint_name}"
    echo "     - KoBEST: ./results/kobest/${WANDB_RUN_ID}-${checkpoint_name}"
    echo "     - HAE-RAE: ./results/haerae/${WANDB_RUN_ID}-${checkpoint_name}"
    echo "     - Interview Eval: ./results/interview_eval/${WANDB_RUN_ID}-${checkpoint_name}"
done
echo ""
echo "🚀 Done!"

deactivate

