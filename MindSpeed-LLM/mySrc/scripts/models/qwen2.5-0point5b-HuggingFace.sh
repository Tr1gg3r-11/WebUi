mkdir -p ${DIR}
cd ${DIR}

wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/config.json
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/generation_config.json
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/merges.txt
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/vocab.json