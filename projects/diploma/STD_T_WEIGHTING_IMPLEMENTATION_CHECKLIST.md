# `std_t` Weighting: Implementation Checklist

## Goal

Добавить в on-policy distillation мягкое взвешивание trajectory-level KL-сигнала по teacher-side Hurst variability:

- считать `std_t = std_k(H_t^k)` по окнам вдоль активной части траектории;
- не оптимизировать `std_t` напрямую;
- использовать `std_t` как вес траектории для KL-части `advantages`.

---

## Recommended First Version

Базовая формула:

- активные teacher logprobs: только позиции с `mask > 0`
- окна фиксированной длины `W = 512`
- на каждом окне считаем `H_t^k`
- затем:

`std_t(τ) = std_k(H_t^k)`

Нормировка внутри батча:

`z_t(τ) = (std_t(τ) - median_batch(std_t)) / (MAD_batch(std_t) + eps)`

Вес траектории:

`w_t(τ) = clip(1 + beta * tanh(z_t(τ)), weight_min, weight_max)`

И затем весим только KL-часть:

`A'_KL(τ) = w_t(τ) * A_KL(τ)`

Итоговый `advantages`:

`A_total(τ) = A_env(τ) + A'_KL(τ)`

---

## Uneven Windows

### Main rule

Окна нужно строить **не по исходным позициям токенов**, а по уже отфильтрованной активной последовательности:

- берем `logprobs[mask > 0]`
- получаем сжатую последовательность длины `N_active`
- режем ее на окна по `W = 512`

Поэтому удаление системного промпта само по себе не создает методологической проблемы: оно просто уменьшает `N_active`.

### What to do with the last partial window

Для первого онлайн-эксперимента рекомендован самый консервативный вариант:

- использовать только **полные окна**
- неполный хвост `< W` просто отбрасывать

То есть:

- `K = floor(N_active / W)`
- используем только первые `K * W` активных токенов

### Why this is the safest choice

- DFA/Hurst на коротком хвосте менее стабилен, чем на полном окне;
- переменная длина последнего окна делает `std_t` менее сопоставимым между траекториями;
- это минимизирует шум в первом прогоне.

### Minimum number of windows

Нужно ввести порог надежности:

- если `K < hurst_min_windows`, то Hurst-weighting не применяем
- в этом случае ставим `w_t = 1.0`

Для первого прогона:

- `W = 512`
- `hurst_min_windows = 6`

Это означает, что weighting включается только если:

- `N_active >= 6 * 512 = 3072`

### Possible future refinement

Если позже захочется учитывать хвост, можно рассмотреть:

- добавление последнего окна только если его длина не меньше `0.75 * W`
- или отдельную early/late сегментацию

Но для первой версии это делать не стоит.

---

## Files To Change

### 1. Core training logic

File:
[train_on_policy.py](/Users/dudberoll/prj/diploma/tinker-cookbook/tinker_cookbook/distillation/train_on_policy.py)

Что добавить:

- новые поля в `Config`
- helper для windowed Hurst
- helper для robust batch normalization
- новую ветку weighting внутри `incorporate_kl_penalty`
- новые training metrics
- расширение telemetry при желании

### 2. Launch recipe / CLI

File:
[on_policy_distillation.py](/Users/dudberoll/prj/diploma/tinker-cookbook/tinker_cookbook/recipes/distillation/on_policy_distillation.py)

Что добавить:

- новые поля в `CLIConfig`
- прокидывание новых полей в `train_on_policy.Config`
- прокидывание `temperature` в `train_on_policy.Config`
- желательно добавить имя режима в `run_name`

### 3. Optional analysis utilities

Не обязательно для первого запуска, но полезно потом:

- [decode_rollouts_check.py](/Users/dudberoll/prj/diploma/decode_rollouts_check.py)
- [windowed_hurst_analysis.py](/Users/dudberoll/prj/diploma/windowed_hurst_analysis.py)
- [signal_decomposition.py](/Users/dudberoll/prj/diploma/signal_decomposition.py)

Они не нужны для обучения, но полезны для проверки того, что online-метрика считается так же, как offline-идея.

---

## Fields To Add In `train_on_policy.Config`

Минимальный набор:

- `hurst_weight_mode: str = "none"`
- `hurst_window_size: int = 512`
- `hurst_min_windows: int = 6`
- `hurst_weight_beta: float = 0.25`
- `hurst_weight_min: float = 0.9`
- `hurst_weight_max: float = 1.1`
- `hurst_weight_eps: float = 1e-6`
- `hurst_weight_warmup_steps: int = 10`
- `hurst_weight_apply_to: str = "kl_only"`

Поддерживаемые режимы `hurst_weight_mode` в первой версии:

- `none`
- `std_t`
- `std_s`

---

## Fields To Add In `CLIConfig`

В [on_policy_distillation.py](/Users/dudberoll/prj/diploma/tinker-cookbook/tinker_cookbook/recipes/distillation/on_policy_distillation.py) добавить те же поля:

- `hurst_weight_mode`
- `hurst_window_size`
- `hurst_min_windows`
- `hurst_weight_beta`
- `hurst_weight_min`
- `hurst_weight_max`
- `hurst_weight_eps`
- `hurst_weight_warmup_steps`
- `hurst_weight_apply_to`

И отдельно убедиться, что уже существующее поле:

- `temperature`

реально прокидывается в `train_on_policy.Config`.

---

## Helpers To Add In `train_on_policy.py`

### A. Window splitter over active sequence

Нужен helper, который:

- берет одномерный tensor logprobs
- берет `window_size`
- возвращает список полных окон длины `window_size`

Поведение:

- хвост короче `window_size` отбрасывается

### B. Windowed Hurst profile

Нужен helper, который:

- берет активную последовательность logprobs
- строит окна
- считает `H` на каждом окне
- возвращает:
  - список `H^k`
  - `std(H^k)`
  - число окон

### C. Robust batch normalization

Нужен helper для батча:

- `median`
- `MAD`
- fallback на `1.0`, если `MAD` почти нулевой

---

## Where The New Logic Should Live

### Important design choice

Weighting по `std_t` надо делать **внутри `incorporate_kl_penalty`**, а не в `prepare_minibatch`.

Почему:

- `std_t` требует teacher logprobs;
- teacher logprobs появляются только после `compute_logprobs_async`;
- значит здесь уже есть все нужные данные:
  - `sampled_logprobs_D`
  - `teacher_logprobs_D`
  - `mask`

### Recommended flow inside `incorporate_kl_penalty`

1. Для каждого datum получить:
   - `lp_s_active`
   - `lp_t_active`

2. Для каждого datum посчитать:
   - `std_s`
   - `std_t`
   - `num_hurst_windows`

3. Собрать по батчу список сигналов:
   - либо `std_t`
   - либо `std_s`
   - в зависимости от `hurst_weight_mode`

4. Нормировать их по батчу и построить `w`

5. Масштабировать только `kl_advantages`

6. Потом уже складывать:

`advantages = old_advantages + weighted_kl_advantages`

---

## Metrics To Log

В `metrics` добавить минимум:

- `optim/hurst_weight_mean`
- `optim/hurst_weight_median`
- `optim/hurst_weight_std`
- `optim/std_t_mean`
- `optim/std_t_median`
- `optim/std_s_mean`
- `optim/std_s_median`
- `optim/hurst_num_windows_mean`
- `optim/hurst_weight_active_frac`

Где:

- `hurst_weight_active_frac` = доля trajectory в батче, для которых weighting реально применился, а не был заменен на `1.0`

---

## Optional Telemetry Fields

В `telemetry_rollouts.jsonl` можно дополнительно писать:

- `std_t`
- `std_s`
- `hurst_weight`
- `num_hurst_windows`

Это не обязательно для первого запуска, но очень полезно для offline-разбора.

---

## Previous Main On-Policy Run

Опорный baseline, с которым лучше сравнивать:

Run:
[onpolicy-deepmath-g64-gs2-s50-20260304-114700](</Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700>)

Параметры:

- `model_name = Qwen/Qwen3-8B-Base`
- `teacher_model = Qwen/Qwen3-32B`
- `renderer_name = role_colon`
- `groups_per_batch = 64`
- `group_size = 2`
- `max_step = 50`
- `learning_rate = 1e-4`
- `lora_rank = 128`
- `max_tokens = 4096`
- `temperature = 1.0`
- `kl_penalty_coef = 1.0`
- `kl_discount_factor = 0.0`
- `num_substeps = 1`
- `fractal_alpha = 0.0`
- `eval_every = 10`
- `save_every = 10`

Dataset builder details:

- `batch_size = 64`
- `group_size = 2`
- `seed = 0`

Teacher:

- `base_model = Qwen/Qwen3-32B`

---

## Historical Hurst-Bonus Run

Run:
[distill-deepmath-Qwen-Qwen3-8B-Base-128rank-0.0001lr-16batch-2026-03-03-18-58](</Users/dudberoll/tinker-examples/distillation/distill-deepmath-Qwen-Qwen3-8B-Base-128rank-0.0001lr-16batch-2026-03-03-18-58>)

Параметры:

- `teacher_model = Qwen/Qwen3-30B-A3B`
- `groups_per_batch = 16`
- `group_size = 1`
- `max_step = 5`
- `fractal_alpha = 0.1`
- `temperature = 1.0`
- `max_tokens = 4096`
- `kl_penalty_coef = 1.0`
- `learning_rate = 1e-4`
- `lora_rank = 128`

Этот запуск полезен как исторический baseline, но не как главный reference для нового weighting-эксперимента.

---

## Recommended New Runs

### Main experiment

Сохранить все как в `g64-gs2-s50`, но включить:

- `hurst_weight_mode = std_t`
- `hurst_window_size = 512`
- `hurst_min_windows = 6`
- `hurst_weight_beta = 0.25`
- `hurst_weight_min = 0.9`
- `hurst_weight_max = 1.1`
- `hurst_weight_eps = 1e-6`
- `hurst_weight_warmup_steps = 10`
- `hurst_weight_apply_to = kl_only`

### Mandatory ablation

Тот же запуск, но:

- `hurst_weight_mode = std_s`

### Strong baseline

Тот же запуск, но:

- `hurst_weight_mode = none`

---

## First Experimental Matrix

Минимальный пакет:

1. `KL-only`
2. `KL + weight(std_t)`
3. `KL + weight(std_s)`

Если есть бюджет, сделать для `std_t` маленький sweep:

- `beta = 0.1`
- `beta = 0.25`
- `beta = 0.5`

---

## Important Control Choice

Для первого запуска лучше **не менять одновременно температуру**.

Даже если есть подозрение, что `temperature = 1.0` ухудшает качество rollout-ов, первый `std_t`-weighting эксперимент лучше провести:

- на той же температуре,
- с тем же `max_tokens`,
- с тем же teacher,
- с тем же batch regime

Иначе нельзя будет понять, откуда пришел эффект.

Отдельная абляция с `temperature = 0.7` — это уже следующий шаг.

---

## Practical Execution Order

1. Добавить новые поля в `train_on_policy.Config`
2. Добавить новые поля в `CLIConfig`
3. Прокинуть `temperature` из recipe в `train_on_policy.Config`
4. Добавить helper для windowed Hurst
5. Добавить расчет `std_t/std_s` внутри `incorporate_kl_penalty`
6. Добавить weighting только для `kl_advantages`
7. Добавить metrics
8. Запустить:
   - `KL-only`
   - `std_t`
   - `std_s`

---

## Default Decision For The Uneven Tail

Фиксируем для первой версии:

- окна считаются по `active_logprobs`
- размер окна: `512`
- используются только полные окна
- хвост отбрасывается
- если полных окон меньше `6`, weight = `1.0`

Это самый аккуратный и наименее шумный вариант для первого online-эксперимента.
