mkdir -p ${DIR}
cd ${DIR}

wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/generation_config.json
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/merges.txt
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/vocab.json