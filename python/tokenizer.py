import pandas as pd
from pathlib import Path
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

'''создаем свой токенизатор'''

BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"

# Создаем пустой BPE токенизатор
tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=1500,
                     special_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
                     min_frequency=1,
                     show_progress=True
                     )

# Обучаем на наборе данных
csv_files = Path('./dataset_720/transcript').glob('*.csv')

sentences =[]
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    if 'text' in df.columns:
        sentences.extend(df['text'].tolist())

corpus = [re.sub(r'[^\w\s]', '', s).lower() for s in sentences]

tokenizer.train_from_iterator(corpus, trainer=trainer)

# Получаем ID спец-токенов
PAD_TOKEN_ID = tokenizer.token_to_id(PAD_TOKEN)
BOS_TOKEN_ID = tokenizer.token_to_id(BOS_TOKEN)
EOS_TOKEN_ID = tokenizer.token_to_id(EOS_TOKEN)

actual_vocab = tokenizer.get_vocab_size()

print(f"Обучение завершено")
print(f"Реальный vocab_size: {actual_vocab}")
print(f"  PAD={PAD_TOKEN_ID}, BOS={BOS_TOKEN_ID}, EOS={EOS_TOKEN_ID}")

# Проверка на примере
example = corpus[9]
enc = tokenizer.encode(example)
print(f"\nПример: {repr(example[:80])}")
print(f"Токены: {enc.ids}")
print(f"Слова:  {enc.tokens}")

tokenizer.save("tokenizer_lipreading.json")
print("Токенизатор сохранен в файл tokenizer_lipreading.json")


