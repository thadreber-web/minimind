import os
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind model inference and chat")
    parser.add_argument('--load_from', default='model', type=str, help="model load path (model=native torch weights, any other path=transformers format)")
    parser.add_argument('--save_dir', default='out', type=str, help="model weights directory")
    parser.add_argument('--weight', default='full_sft', type=str, help="weight name prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA weight name (None means disabled; options: lora_identity, lora_medical)")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank, must match training (default 16)")
    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="number of hidden layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="use MoE architecture (0=no, 1=yes)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="enable RoPE position-encoding extrapolation (4x; only addresses position encoding)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="max generation length (note: not the model actual long-context ability)")
    parser.add_argument('--temperature', default=0.85, type=float, help="generation temperature controlling randomness (0-1, higher is more random)")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus sampling threshold (0-1)")
    parser.add_argument('--open_thinking', default=0, type=int, help="enable adaptive thinking (0=no, 1=yes)")
    parser.add_argument('--historys', default=0, type=int, help="number of history turns to carry (must be even; 0 means no history)")
    parser.add_argument('--show_speed', default=1, type=int, help="show decode speed (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="device to run on")
    args = parser.parse_args()
    
    prompts = [
        'What are your strengths?',
        'Why is the sky blue?',
        'Write a Python function that computes the Fibonacci sequence.',
        'Explain the basic process of "photosynthesis".',
        'If it rains tomorrow, how should I get ready to go out?',
        'Compare the pros and cons of cats and dogs as pets.',
        'Explain what machine learning is.',
        'Recommend some famous dishes from around the world.'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] Automatic test\n[1] Manual input\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()