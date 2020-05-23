# Restraining-Bolts-for-Reinforcement-Learning


## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
    
    
# Training

```
$ python3 train.py
    --episodes 10000
    --env
    --algo dqn
    --gui False
    --model_name
    --rand_seed 42
```

# Testing

```
$ python3 test.py
    --model_name
    --record True
```
If you want to record, please remember to install ffmpeg

On linux:
```
sudo apt-get install ffmpeg
```

On ubuntu:
```
brew install ffmpeg
```

On Windows:
good luck
