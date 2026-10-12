# Experiment Configs Summary

Дата составления: 2026-06-09

Этот файл фиксирует основные эксперименты по on-policy distillation, Hurst/std_t regularization, Golden Zone baseline и eval baseline. Точные параметры каждого запуска сохранены в соответствующих `config.json`; здесь собрана удобная master-сводка для анализа, доклада и диплома.

## Общая постановка

Базовый student:

`Qwen/Qwen3-8B-Base`

Основной teacher:

`Qwen/Qwen3-32B`

Dataset:

`deepmath`

Renderer:

`role_colon`

Базовая training-схема:

- on-policy distillation;
- student генерирует rollout;
- teacher оценивает student-generated tokens через logprobs;
- token-level reverse KL добавляется в `advantages`;
- дополнительные Hurst/GZ-сигналы являются trajectory-level модификациями.

Ключевой frozen eval subset:

`/Users/dudberoll/prj/diploma/eval_subsets/deepmath_eval_512_seed0.jsonl`

Обозначения для объема обучения:

- `groups_per_batch` - число prompt-групп на training step;
- `group_size` - число rollout-ов на один prompt;
- `rollouts_per_step = groups_per_batch * group_size`;
- `total_rollouts = rollouts_per_step * max_step`.

## 1. Base Model Eval: Qwen3-8B-Base, No Training

Цель:

Получить чистый baseline качества без on-policy дообучения на фиксированном eval-наборе.

Артефакты:

- subset: `/Users/dudberoll/prj/diploma/eval_subsets/deepmath_eval_512_seed0.jsonl`
- results: `/Users/dudberoll/prj/diploma/eval_runs/qwen3_8b_base_deepmath_512_seed0/results.jsonl`
- summary: `/Users/dudberoll/prj/diploma/eval_runs/qwen3_8b_base_deepmath_512_seed0/summary.json`

Параметры:

| Parameter | Value |
|---|---:|
| model | `Qwen/Qwen3-8B-Base` |
| subset size | `512` |
| seed | `0` |
| max_tokens | `4096` |
| temperature | `1.0` |
| concurrency | `8` |

Итог:

| Metric | Value |
|---|---:|
| correct_mean | `0.458984` |
| reward_mean | `0.358984` |
| format_mean | `0.0` |
| parse_success_mean | `0.0` |
| boxed_rate | `0.90625` |
| response_tokens_mean | `1157.86` |
| response_tokens_median | `759` |
| maxed_out_rate | `0.05469` |

Вывод:

Base model уже дает около `45.9%` correct на frozen DeepMath-512 и редко упирается в `4096` токенов. Это главный fixed baseline для дальнейшего сравнения.

## 2. Legacy KL-only On-policy Run: g64-gs2-s50

Цель:

Исторический основной on-policy запуск, близкий к исходной постановке. Важен как reference run для анализа длины, деградации и checkpoint loading.

Артефакты:

- run: `/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700`
- config: `/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700/config.json`
- metrics: `/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700/metrics.jsonl`
- telemetry: `/Users/dudberoll/tinker-examples/distillation/onpolicy-deepmath-g64-gs2-s50-20260304-114700/telemetry_rollouts.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| model_name | `Qwen/Qwen3-8B-Base` |
| load_checkpoint_path | `tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final` |
| teacher_model | `Qwen/Qwen3-32B` |
| groups_per_batch | `64` |
| group_size | `2` |
| trajectories per step | `128` |
| max_step | `50` |
| total rollouts | `6400` |
| learning_rate | `1e-4` |
| lora_rank | `128` |
| max_tokens | `4096` |
| temperature | `1.0` |
| kl_penalty_coef | `1.0` |
| fractal_alpha | `0.0` |
| eval_every | `10` |
| save_every | `10` |

Final checkpoint:

- state: `tinker://dd732f19-119b-5d0c-bed5-89962be0d690:train:0/weights/final`
- sampler: `tinker://dd732f19-119b-5d0c-bed5-89962be0d690:train:0/sampler_weights/final`

Final train metrics:

| Metric | Value |
|---|---:|
| final step | `49` |
| correct | `0.164063` |
| reward | `0.064063` |
| teacher_kl | `0.197352` |
| ac_tokens_per_turn | `4096.0` |
| h_student | `0.665078` |
| h_teacher | `0.650605` |
| fractal_dissonance | `0.024397` |

Вывод:

К финалу запуск сильно ушел в max-length режим: `ac_tokens_per_turn = 4096`. Ранее по кривой были лучшие шаги, поэтому финальная метрика не отражает лучший checkpoint.

## 3. KL-only Budget Run: g48-gs2-s24

Цель:

Бюджетный KL-only контроль того же масштаба, что и новые регуляризаторы.

Артефакты:

- run: `/Users/dudberoll/prj/diploma/train_runs/klonly_g48_gs2_s24_20260512`
- config: `/Users/dudberoll/prj/diploma/train_runs/klonly_g48_gs2_s24_20260512/config.json`
- metrics: `/Users/dudberoll/prj/diploma/train_runs/klonly_g48_gs2_s24_20260512/metrics.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| model_name | `Qwen/Qwen3-8B-Base` |
| load_checkpoint_path | `null` |
| teacher_model | `Qwen/Qwen3-32B` |
| groups_per_batch | `48` |
| group_size | `2` |
| trajectories per step | `96` |
| max_step | `24` |
| total rollouts | `2304` |
| learning_rate | `1e-4` |
| lora_rank | `128` |
| max_tokens | `4096` |
| temperature | `1.0` |
| kl_penalty_coef | `1.0` |
| fractal_alpha | `0.0` |
| hurst_weight_mode | `none` |

Final checkpoint:

- state: `tinker://93970833-ca80-57cd-9065-93f353996000:train:0/weights/final`
- sampler: `tinker://93970833-ca80-57cd-9065-93f353996000:train:0/sampler_weights/final`

Final train metrics:

| Metric | Value |
|---|---:|
| final step | `23` |
| correct | `0.291667` |
| reward | `0.19375` |
| teacher_kl | `0.245879` |
| ac_tokens_per_turn | `3808.625` |
| h_student | `0.680603` |
| h_teacher | `0.673584` |
| std_t_mean | `0.079534` |
| std_s_mean | `0.081557` |

Вывод:

Даже KL-only budget run к финалу удлиняет ответы до `~3809` токенов и проседает по quality. Это важный контроль: часть деградации связана не только с регуляризаторами, но и с самим режимом on-policy обучения.

## 4. std_t Weighting, Fixed Windows, min_windows=6

Цель:

Первая попытка использовать teacher-side Hurst variability как weighting signal. Сигнал должен был усиливать KL на траекториях с высоким `std_t`.

Артефакты:

- run: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_20260512`
- config: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_20260512/config.json`
- metrics: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_20260512/metrics.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| groups_per_batch | `48` |
| group_size | `2` |
| rollouts_per_step | `96` |
| max_step | `24` |
| total rollouts if completed | `2304` |
| hurst_weight_mode | `std_t` |
| hurst_window_size | `512` |
| hurst_min_windows | `6` |
| hurst_weight_beta | `0.5` |
| hurst_weight_min | `0.8` |
| hurst_weight_max | `1.2` |
| hurst_weight_warmup_steps | `5` |

Observed behavior:

- run was stopped early after `4` logged steps;
- completed rollouts were approximately `384`;
- `hurst_weight_enabled = 0.0` during logged steps because warmup and `min_windows=6` prevented activation;
- average number of windows was only about `2.0-2.35`.

Вывод:

С `min_windows=6` weighting почти не включался на ранних trajectories. Это привело к следующему эксперименту с `min_windows=2`.

## 5. std_t Weighting, Fixed Windows, min_windows=2

Цель:

Сделать `std_t` weighting реально активным и проверить, улучшает ли он on-policy distillation.

Артефакты:

- run: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_mw2_20260512`
- config: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_mw2_20260512/config.json`
- metrics: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_b05_mw2_20260512/metrics.jsonl`
- interrupted eval: `/Users/dudberoll/prj/diploma/eval_runs/stdt_final_deepmath_512_seed0/results.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| groups_per_batch | `48` |
| group_size | `2` |
| rollouts_per_step | `96` |
| max_step | `24` |
| total rollouts | `2304` |
| hurst_weight_mode | `std_t` |
| hurst_window_size | `512` |
| hurst_min_windows | `2` |
| hurst_weight_beta | `0.5` |
| hurst_weight_min | `0.8` |
| hurst_weight_max | `1.2` |
| hurst_weight_warmup_steps | `0` |

Final checkpoint:

- state: `tinker://2e61ec7a-fce7-5b3d-929e-d18959fa6480:train:0/weights/final`
- sampler: `tinker://2e61ec7a-fce7-5b3d-929e-d18959fa6480:train:0/sampler_weights/final`

Final train metrics:

| Metric | Value |
|---|---:|
| final step | `23` |
| correct | `0.25` |
| reward | `0.15` |
| teacher_kl | `0.24443` |
| ac_tokens_per_turn | `3593.135` |
| hurst_weight_active_frac | `0.958333` |
| hurst_weight_std | `0.17864` |
| std_t_mean | `0.066513` |
| std_s_mean | `0.069424` |

Interrupted frozen eval:

| Metric | Value |
|---|---:|
| evaluated examples | `75 / 512` |
| correct_mean | `0.213333` |
| reward_mean | `0.113333` |
| response_tokens_mean | `3943.59` |
| response_tokens_median | `4096` |
| maxed_out_rate | `0.893333` |

Вывод:

`std_t` weighting стал активным, но вызвал сильный length drift. Модель начала уходить в длинные ответы, часто до `4096`, а качество упало. Это главный отрицательный результат по `std_t`: offline-сигнал был информативным, но online-weighting без контроля длины привел к metric hacking через длину траектории.

## 6. std_t Weighting, Adaptive 10 Windows + Length Gate

Цель:

Исправить проблему length drift: использовать `std_t` только в здоровой длиновой зоне и считать Hurst по `10` adaptive windows.

Артефакты:

- run: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_adaptive10_b025_20260512`
- config: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_adaptive10_b025_20260512/config.json`
- metrics: `/Users/dudberoll/prj/diploma/train_runs/stdt_g48_gs2_s24_adaptive10_b025_20260512/metrics.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| groups_per_batch | `48` |
| group_size | `2` |
| rollouts_per_step | `96` |
| max_step | `24` |
| total rollouts | `2304` |
| hurst_weight_mode | `std_t` |
| hurst_num_windows | `10` |
| hurst_n_active_min | `1024` |
| hurst_n_active_max | `1536` |
| hurst_weight_beta | `0.25` |
| hurst_weight_min | `0.9` |
| hurst_weight_max | `1.1` |
| hurst_weight_warmup_steps | `5` |

Final checkpoint:

- state: `tinker://b18a1055-7ea9-5d46-a97c-23ccd5055765:train:0/weights/final`
- sampler: `tinker://b18a1055-7ea9-5d46-a97c-23ccd5055765:train:0/sampler_weights/final`

Final train metrics:

| Metric | Value |
|---|---:|
| final step | `23` |
| correct | `0.260417` |
| reward | `0.1625` |
| teacher_kl | `0.259737` |
| ac_tokens_per_turn | `3558.021` |
| hurst_weight_active_frac | `0.020833` |
| hurst_weight_std | `0.014434` |
| h_student | `0.674778` |
| h_teacher | `0.665329` |
| std_t_mean | `0.082889` |
| std_s_mean | `0.084035` |

Вывод:

Length gate почти выключил weighting к финалу, потому что trajectories снова стали слишком длинными. Деградация длины все равно произошла. Эта версия безопаснее фиксированного `min_windows=2`, но не решила проблему on-policy length drift.

## 7. Golden Zone Hurst Baseline: GZ-H

Цель:

Проверить более консервативный regularizer: не поощрять рост Hurst и не делать teacher matching, а удерживать student Hurst внутри healthy zone.

Формула:

```text
P_gz(H_s) = max(0, |H_s - 0.70| - 0.09)^2
A'_t = A_t - lambda_gz * P_gz(H_s)
```

Артефакты:

- code copy: `/Users/dudberoll/prj/diploma/tinker-cookbook-reg2`
- run: `/Users/dudberoll/prj/diploma/tinker-cookbook-reg2/train_runs/gzh_g48_gs2_s24_l003_20260512`
- config: `/Users/dudberoll/prj/diploma/tinker-cookbook-reg2/train_runs/gzh_g48_gs2_s24_l003_20260512/config.json`
- metrics: `/Users/dudberoll/prj/diploma/tinker-cookbook-reg2/train_runs/gzh_g48_gs2_s24_l003_20260512/metrics.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| groups_per_batch | `48` |
| group_size | `2` |
| rollouts_per_step | `96` |
| max_step | `24` |
| total rollouts | `2304` |
| hurst_weight_mode | `none` |
| golden_zone_mode | `h` |
| golden_zone_lambda | `0.03` |
| golden_zone_h_target | `0.70` |
| golden_zone_h_delta | `0.09` |

Final checkpoint:

- state: `tinker://c671e4c4-6e41-59ab-9f42-3698b71bdbe9:train:0/weights/final`
- sampler: `tinker://c671e4c4-6e41-59ab-9f42-3698b71bdbe9:train:0/sampler_weights/final`

Final train metrics:

| Metric | Value |
|---|---:|
| final step | `23` |
| correct | `0.208333` |
| reward | `0.111458` |
| teacher_kl | `0.248917` |
| ac_tokens_per_turn | `3777.531` |
| h_student | `0.674386` |
| gz_penalty_mean | `0.000584` |
| gz_active_frac | `0.302083` |
| gz_h_excess | `0.010574` |

Вывод:

GZ-H технически работал и penalty был корректно залогирован, но с `lambda=0.03` не предотвратил length drift и не улучшил train quality. Как baseline он полезен: показывает, что мягкий Hurst-only healthy-zone штраф сам по себе недостаточен в текущем on-policy режиме.

## 8. Checkpoint Load Smoke: External Tinker State

Цель:

Проверить, можно ли использовать внешний Tinker checkpoint как `load_checkpoint_path` для `on_policy_distillation`.

Артефакты:

- run: `/Users/dudberoll/prj/diploma/test_runs/load_ckpt_smoke_20260514`
- config: `/Users/dudberoll/prj/diploma/test_runs/load_ckpt_smoke_20260514/config.json`
- metrics: `/Users/dudberoll/prj/diploma/test_runs/load_ckpt_smoke_20260514/metrics.jsonl`

Параметры:

| Parameter | Value |
|---|---:|
| model_name | `Qwen/Qwen3-8B-Base` |
| load_checkpoint_path | `tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final` |
| teacher_model | `Qwen/Qwen3-32B` |
| groups_per_batch | `1` |
| group_size | `1` |
| rollouts_per_step | `1` |
| max_step | `1` |
| total rollouts | `1` |
| max_tokens | `512` |
| hurst_weight_mode | `none` |

Result:

- run reached sampling, KL computation, train, telemetry writing, and checkpoint saving;
- therefore this `load_checkpoint_path` is valid for Tinker training resume/init.

Created checkpoint:

- state: `tinker://7825283e-1283-570d-9194-b8a790ba3bbf:train:0/weights/final`
- sampler: `tinker://7825283e-1283-570d-9194-b8a790ba3bbf:train:0/sampler_weights/final`

## Offline Analysis Reports

Важные отчеты, которые объясняют выбор регуляризаторов:

- `/Users/dudberoll/prj/diploma/offline_validation_results/windowed_hurst_report.md`
- `/Users/dudberoll/prj/diploma/offline_validation_results/signal_decomposition_report.md`
- `/Users/dudberoll/prj/diploma/offline_validation_results/confounder_report.md`
- `/Users/dudberoll/prj/diploma/offline_validation_results/temporal_hurst_report.md`
- `/Users/dudberoll/prj/diploma/offline_validation_results/hurst_length_report.md`
- `/Users/dudberoll/prj/diploma/HURST_REGULARIZATION_NEXT_STEPS.md`
- `/Users/dudberoll/prj/diploma/STD_T_WEIGHTING_IMPLEMENTATION_CHECKLIST.md`
- `/Users/dudberoll/prj/diploma/HURST_ESTIMATION_DESCRIPTION.md`

Ключевые offline-выводы:

- global `|H_s - H_t|` на всей последовательности слабый;
- windowed `std(H_s^k - H_t^k)` заметно сильнее global Hurst-diff;
- затем decomposed analysis показал, что самый сильный сигнал был не `std_diff`, а `std_t = std(H_t^k)` и `std_s = std(H_s^k)`;
- `std_t` лучше интерпретировать как teacher-side detector структуры траектории, а не как alignment objective;
- online-оптимизация/weighting этого сигнала оказалась чувствительной к длине ответа.

## Сравнение Основных Запусков

| Experiment | Main signal | Steps | Traj/step | Final correct | Final reward | Final tokens/turn | Key outcome |
|---|---|---:|---:|---:|---:|---:|---|
| Base eval | none | eval only | - | `0.459` | `0.359` | `1158 mean` | strong no-train baseline |
| Legacy KL-only | KL | `50` | `128` | `0.164` | `0.064` | `4096` | `6400` rollouts, final checkpoint maxed out |
| KL-only budget | KL | `24` | `96` | `0.292` | `0.194` | `3809` | `2304` rollouts, length drift even without Hurst |
| std_t fixed min=2 | KL weighted by std_t | `24` | `96` | `0.25` | `0.15` | `3593` | `2304` rollouts, active weighting, degraded quality |
| std_t adaptive gate | gated KL weighting | `24` | `96` | `0.260` | `0.163` | `3558` | `2304` rollouts, gate reduced active weighting, drift remained |
| GZ-H | Hurst zone penalty | `24` | `96` | `0.208` | `0.111` | `3778` | `2304` rollouts, penalty logged, did not stop drift |

## Current Interpretation

1. Base model evaluation is surprisingly strong relative to trained final checkpoints.
2. On-policy training in this setup tends to increase response length toward `4096`.
3. `std_t` is useful as an offline diagnostic but risky as an online weighting signal because it is entangled with trajectory length.
4. Adaptive windows and length gates reduce direct activation of `std_t`, but do not solve length drift by themselves.
5. GZ-H is a clean baseline and scientifically defensible, but the tested strength `lambda=0.03` was too weak or too indirect to stabilize generation.
6. The next serious experiments should control length explicitly, lower temperature, use an SFT/init checkpoint, or add a direct completion-length/stop-format mechanism before relying on Hurst-based regularization.
