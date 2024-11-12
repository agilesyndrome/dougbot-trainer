from lib import TokenizerUtils
from lib import Config
from lib import HuggingFaceDataset
from lib import LLM

cfg = Config()
ds = HuggingFaceDataset(cfg)

llm = LLM(cfg)

token_helper = TokenizerUtils(llm.tokenizer)
token_helper.process(ds)

print(ds.ds)

llm.train(ds)
