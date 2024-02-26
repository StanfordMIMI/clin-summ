conda create -n clin_summ_tim python=3.9
conda activate clin_summ_tim
conda install pip
conda install transformers
conda install -y numpy pandas
conda install -y matplotlib pillow
pip install torch torchaudio torchvision
pip install peft
pip install auto-gptq
pip install tiktoken==0.1.1
pip install evaluate
pip install tensorboard
pip install f1chexbert
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install radgraph
pip install rouge_score