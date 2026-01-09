# featurize dataset extract a8afcc12-1cb6-42d0-a163-91d9d1c72399 ~/data
# hf download lightx2v/Wan2.1-Distill-Models wan2.1_i2v_480p_lightx2v_4step.safetensors --local-dir /home/featurize/data/fast-multitalk
# hf download MeiGen-AI/MeiGen-MultiTalk multitalk.safetensors --local-dir /home/featurize/data/fast-multitalk  
pip install transformers --upgrade
pip uninstall tensorflow -y
conda install ffmpeg -y
pip install num2words -i https://pypi.org/simple
pip install -r requirements.txt
pip install /home/featurize/work/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
hf download Ddream-ai/fast-multitalk --local-dir ~/data/fast-multitalk
