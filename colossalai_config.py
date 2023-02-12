from colossalai.amp import AMP_TYPE
import os

from colossalai.zero.shard_utils import TensorShardStrategy

CONFIG = dict(fp16=dict(mode=AMP_TYPE.NAIVE))
