# Анализ обучения: 2D CNN + взвешенный loss (v2)

## Результаты эксперимента

| Метрика | Значение |
|---|---|
| Эпох обучено | 66+ |
| Best val_loss | ~7.51 (эпоха 42) |
| Test loss | 7.55 |
| Test token accuracy | **11.8%** |
| Предсказание на всех тестовых примерах | `""` (пустая строка) |
| train_acc после 66 эпох | **0.108 — не изменился с эпохи 8** |

---

## Диагноз: три независимые проблемы

### 1. Ошибочный вес токена EOS

Вес токена вычислялся по `data["input_ids"]`, куда EOS **не попадает** — это
специальный токен, добавляемый только в `__getitem__`. Итог:

| Токен | Частота в счётчике | Вес в loss |
|---|---|---|
| `"."` (самый частый) | 269 | **0.022** |
| EOS (id=2) | 0 | **1.0** (дефолт) |

Вес EOS в **45 раз** выше веса `"."`. Модель получает огромный градиент за
предсказание EOS и быстро находит вырожденное решение:
**предсказывать EOS на каждой позиции** (пустая строка на выходе).

### 2. train_acc = 0.108 с эпохи 8 — модель не обучается даже на train

При teacher forcing декодер получает правильные предыдущие токены. Даже в этих
условиях модель не может поднять train_acc выше 10.8%. Это означает, что
**нет разрыва между train_acc и val_acc** — классический признак того, что
модель не учится вообще, а не просто переобучается.

Причина: декодер нашёл тривиальное решение (EOS везде), и градиенты больше
не обновляют веса энкодера — он фактически отключён.

### 3. Label smoothing конфликтует с взвешенным loss

`label_smoothing=0.1` равномерно распределяет 10% вероятности на все токены
**включая PAD** (который `ignore_index` убирает из loss, но не из сглаживания).
При взвешенном loss это создаёт нестабильный градиентный сигнал: сглаживание
«тянет» в одну сторону, веса — в другую.

---

## Анализ: почему seq2seq не обучается на 15 клипах

Seq2seq с cross-attention требует одновременного обучения трёх вещей:
1. Извлечение визуальных признаков (CNN)
2. Выравнивание видео ↔ текст (cross-attention)
3. Языковая модель (decoder self-attention)

На 15 клипах модель не может решить все три задачи одновременно — она бросает (1)
и (2), оставляя только (3) в виде тривиальной стратегии (EOS везде).

**CTC (Connectionist Temporal Classification) требует только (1) + частичную (3):**
выравнивание учится неявно, без cross-attention. Это принципиально более
простая задача для малых датасетов.

---

## Предложения по улучшению

### Исправление 1: правильные веса EOS/BOS (быстро, обязательно)

```python
# После подсчёта token_counts добавить:
avg_weight = token_weights[token_weights > 0].mean().item()
token_weights[EOS_TOKEN_ID] = avg_weight   # EOS — редкий, но не «бесконечно ценный»
token_weights[BOS_TOKEN_ID] = 0.0          # BOS никогда не является таргетом
```

### Исправление 2: убрать label_smoothing при взвешенном loss

```python
criterion = nn.CrossEntropyLoss(
    weight=token_weights.to(DEVICE),
    ignore_index=PAD_TOKEN_ID,
    label_smoothing=0.0,   # убрать — конфликтует с весами
)
```

### Диагностика 3: overfit на одном батче

Если модель не может выучить один батч за 200 шагов — баг в архитектуре или
потоке данных.

```python
# Добавить ячейку перед основным циклом:
model_test = LipReadingTransformer2D(MODEL_CONFIG).to(DEVICE)
opt_test   = torch.optim.Adam(model_test.parameters(), lr=1e-3)
batch_test = next(iter(train_loader))
frames_t   = batch_test["frames"].to(DEVICE)
src_m_t    = batch_test["src_padding_mask"].to(DEVICE)
dec_t      = batch_test["decoder_input"].to(DEVICE)
tgt_t      = batch_test["target"].to(DEVICE)
tgt_m_t    = batch_test["tgt_padding_mask"].to(DEVICE)

for step in range(300):
    logits = model_test(frames_t, dec_t, tgt_padding_mask=tgt_m_t, src_padding_mask=src_m_t)
    loss   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(logits.reshape(-1, VOCAB_SIZE), tgt_t.reshape(-1))
    opt_test.zero_grad(); loss.backward(); opt_test.step()
    if step % 50 == 0:
        acc = (logits.argmax(-1)[tgt_t != PAD_TOKEN_ID] == tgt_t[tgt_t != PAD_TOKEN_ID]).float().mean()
        print(f"Step {step:3d} | loss={loss.item():.4f} | acc={acc.item():.3f}")
```

Ожидаемый результат: acc → 0.9+ за 200–300 шагов. Если нет — архитектурная
проблема.

### Диагностика 4: мониторинг градиентов по компонентам

```python
# Добавить в train_one_epoch после loss.backward():
enc_cnn_norm = sum(p.grad.norm().item()**2
                   for p in model.encoder.cnn_backbone.parameters()
                   if p.grad is not None) ** 0.5
enc_tr_norm  = sum(p.grad.norm().item()**2
                   for p in model.encoder.transformer_encoder.parameters()
                   if p.grad is not None) ** 0.5
dec_norm     = sum(p.grad.norm().item()**2
                   for p in model.decoder.parameters()
                   if p.grad is not None) ** 0.5
# Если enc_cnn_norm ≈ 0 → encoder не участвует в обучении
```

### Архитектурная альтернатива 5: CTC вместо seq2seq

CTC убирает необходимость в cross-attention и явном выравнивании.
Энкодер напрямую выдаёт распределение по токенам для каждого кадра:

```
VisualEncoder2D → [B, T, d_model] → Linear(d_model, vocab_size+1) → CTCLoss
```

**Плюсы для нашего случая:**
- Нет decoder, нет cross-attention — меньше компонентов, которые нужно
  обучать одновременно
- CTC работает на малых датасетах (используется в speech recognition с нуля)
- Не нужен max_tokens — выход имеет длину T (число кадров)
- Инференс проще: beam search или greedy по кадрам

**Минусы:**
- Требует, чтобы T ≥ длина целевой последовательности (40 кадров >> 11 токенов ✓)
- Токенизация должна быть на уровне символов или коротких подслов
  (BPE с vocab=1200 подходит)

---

## Рекомендуемый порядок действий

| Шаг | Действие | Цель |
|---|---|---|
| 1 | Исправить вес EOS/BOS | устранить ложный градиентный стимул |
| 2 | Убрать label_smoothing | убрать конфликт с весами |
| 3 | Запустить overfit-тест на 1 батче | подтвердить, что архитектура работает |
| 4 | Проверить grad norm по компонентам | убедиться, что encoder получает градиент |
| 5 | Если (3) проходит — перезапустить обучение | возможно, достаточно фиксов 1+2 |
| 6 | Если (3) не проходит — перейти на CTC | архитектурная проблема |
