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
- 由于Google官方的mT5模型并没有中日支持（虽然model card上有写，但是实际没有），所以在训练中日语言模型时可以使用我的版本的[预训练模型](https://drive.google.com/file/d/1rJWMnt6n_yU23MMQTABPXahVGTSFVhSS/view?usp=share_link)  
  - *NOTE:预训练模型使用的是mT5-base, 如果要使用small或者large等其他预训练模型，只需要把下下来的预训练模型文件夹里的pytorch_model.bin和config.json(不是config文件夹下的config.json)替换掉就行，不要替换spiece.model和tokenizer.json*  

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
