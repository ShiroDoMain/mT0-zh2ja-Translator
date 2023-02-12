此模型使用了google的mT5多语言模型，用于训练中文到日语的翻译或者日语到中文的翻译  
# Usage  
## train
- 根据config/config.json可以调整模型训练的参数，目前fp16训练有loss nan的问题  
- colossalai的参数可以在colossalai_config.py里调整， 注意：*epoch和batch依然是使用config.json的参数，在colossalai_config.py中设置的全局参数不会起作用*  
- load_epoch可以用于恢复之前训练的checkpoint  
- 数据文件应放在 *data/* 目录下，以源语言/目标语言为后缀保存
```
...
├───data
│   └───train.ja
│   └───train.zh
│   └───val.ja
│   └───val.zh
│   ...
...
 
```
训练运行脚本
```bash
python train.py
```
## inference  
- 推理时需要在config/config.json里指定load_epoch,可以是指定的已经训练完成的epoch，也可以是已保存的最好的模型  
- 推理时应该使用cpu推理  
- 推理时会自动加上前缀  
```
{
    ...
    "load_epoch": "best" or num,
    ...
    "device": "cpu"
}
```   
```bash
python inference.py
```
