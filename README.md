# steam_audio — Steam Audio для Horizon

Нативный мод (`.hmod`) для [Horizon](https://github.com/horizon-rust/horizon), который заменяет стереопанораму
Horizon (копию OpenAL Soft) на **бинауральный рендер Steam Audio** из
[steam-audio-rust](https://github.com/razedproduction/steam-audio-rust) — Rust-порта Valve Steam Audio.

- **HRTF**: каждый звук в мире проходит через свёртку со встроенным HRTF Steam Audio (CIPIC, субъект 124,
  1250 направлений). В наушниках слышно, где источник — сзади, сверху или снизу, а не только слева/справа.
- **Поглощение в воздухе**: трёхполосная модель Steam Audio по умолчанию (`AirAbsorptionModel`): дальние звуки
  теряют высокие частоты.
- **Геймплей не меняется**: затухание по расстоянию остаётся ванильным (линейное), поэтому что слышно и
  насколько громко — как без мода; меняется только то, *как* это звучит.
- Музыка, интерфейс и звуки «у слушателя» без направления панорамируются как в Horizon.

## Совместимость

| | |
| --- | --- |
| Horizon | Nightly 2026-11-04, `6475f62f3c834b7489e1816bf47f47ea5eb2b879` (`nightly-source/2026-11-04-6475f62`) |
| horizon-mixins | `e8cd0aa38cffef49cb2e088ef716baa107148c02` (ревизия лаунчера на этом коммите) |
| Minecraft | 26.3 |
| Частота вывода | 44.1, 48 или 24 кГц (частоты встроенного HRTF); на других звук остаётся ванильным |

Мод привязан к точному коммиту Horizon: он трогает внутренности `horizon-client-render`, а не стабильные
точки `native_hooks`, так что на другом Nightly лаунчер предупредит о несовпадении версий.

## Установка

Соберите файл (нужен Python 3.11+):

```bash
python tools/package.py
```

Получится `dist/steam_audio-0.1.0.hmod`: `horizon-native.json` и `mixins/steam_audio/`. Реестр Horizon пока
публикует нативные моды как `.hnmod` (тот же формат); `--hnmod` положит рядом копию с этим расширением.

Собрать игру с модом через лаунчер, указав папку бандла (или распакованный `.hmod`):

```bash
horizon-launcher native-build mixins/steam_audio --trust-native-code
```

Нативные моды — это код с правами пользователя; лаунчер компилирует его в игру. Прочитайте исходники перед
установкой (они все в `mixins/steam_audio/`).

## Настройка

Через переменные окружения игры:

| Переменная | Что делает |
| --- | --- |
| `HORIZON_STEAM_AUDIO=0` | выключить мод: звук как в Horizon |
| `HORIZON_STEAM_AUDIO_AIR=0` | выключить поглощение в воздухе |
| `HORIZON_STEAM_AUDIO_INTERPOLATION=bilinear` | билинейная интерполяция HRTF (плавнее при движении, дороже) |
| `HORIZON_STEAM_AUDIO_VOICES=N` | сколько самых громких источников рендерить бинаурально (по умолчанию 32; остальные — ванильная панорама) |
| `HORIZON_STEAM_AUDIO_VOLUME_DB=x` | поправка громкости HRTF в дБ (−24…24) |
| `HORIZON_STEAM_AUDIO_HRTF=путь` | свой HRTF в формате Steam Audio `.hrtf` вместо встроенного |

Бинауральный звук рассчитан на наушники.

## Как это устроено

```
mixins/steam_audio/
  horizon-mixin.toml   манифест: модуль, хук, крейт
  listener.rs          HEAD-хук на Audio::set_listener
  output.rs            support-модуль crate::audio::steam_audio (внутри audio.rs Horizon)
  engine/              крейт steam-audio-hmod: вендоренный steam-audio-rust + рендерер
    data/cipic_124.hrtf встроенный HRTF (tools/gen_hrtf.py из core/src/core/cipic_124.inl)
```

Horizon играет звук одним cpal-потоком, колбэк которого вызывает `audio::mix`. Перехватить сам `mix`
horizon-mixins не даёт (`E_UNSUPPORTED_CFG` из‑за `#[cfg_attr(...)]` на функции, а вызывающий её
`build_stream` целиком под `#[cfg(feature = "audio")]`). Поэтому мод **берёт вывод на себя**:

1. При обновлении камеры (`Audio::set_listener`) хук вызывает `steam_audio::take_over`. Тот открывает
   собственный поток на том же устройстве и ставит поток Horizon на **паузу** (не закрывает: Horizon по-прежнему
   видит открытый вывод и продолжает следить за сменой устройств).
2. Колбэк мода блокирует тот же `Mixer` и рендерит его через `steam_audio_hmod::horizon::Renderer`: та же
   геометрия, что у ванильного микшера (базис камеры, проверка мира, затухание), затем air absorption (`EqEffect`)
   и `BinauralEffect` Steam Audio кадрами по 512 сэмплов, и ванильный лимитер.
3. Пока HRTF грузится (в отдельном потоке, доли секунды), на неподдерживаемой частоте, моно-выходе и в
   splitscreen колбэк вызывает ванильный `mix` — звук не пропадает ни на миг.
4. Ванильный `mix` заполняет `Mixer::cameras`, а вывод мода оставляет его пустым. Заполненный — значит, поток
   Horizon снова играет (лаунчер переоткрыл устройство): мод переоткрывает свой и снова ставит тот на паузу.

### Отличия вендоренного steam-audio-rust

Крейт мода может зависеть только от крейтов, уже записанных в `Cargo.lock` Horizon. `rustfft`, `realfft` и
`num-complex` там нет, поэтому:

- `src/fft.rs` — собственное БПФ (итеративный radix-2, real-to-complex через комплексное БПФ половинного
  размера, поворотные множители в `f64`) с тем же API;
- `src/complex.rs` — свой `Complex32` с нужным подмножеством API `num_complex`;
- встроенный HRTF читается из `engine/data/` вместо `build.rs`; поддержка SOFA не компилируется;
- `src/horizon.rs` — новый: рендерер для микшера Horizon.

Остальной код steam-audio-rust не изменён.

## Проверка

```bash
cargo test -p steam-audio-hmod
```

128 тестов: весь набор юнит-тестов steam-audio-rust плюс тесты БПФ (сравнение с ДПФ в `f64`) и рендерера.
Кроме того, при разработке:

- golden-тесты оригинального steam-audio-rust (сверка с эталонными данными из C++ Steam Audio: свёртка, HRTF,
  spatialization, reverb, симуляция…) прогнаны с этим БПФ и `Complex32` — все проходят;
- `output.rs` и `listener.rs` скомпилированы и протестированы в mock-крейте с точными копиями `Mixer`,
  `Channel`, `Audio`, `mix`, `stereo_gains` из `audio.rs` закреплённого Nightly, с cpal 0.16.0 и без фичи
  `audio`: источник справа от игрока громче в правом ухе (в том числе после поворота), откат на `mix` в
  splitscreen, остановка доигравших звуков;
- нагрузка: 32 бинауральных голоса одновременно — около 15% одного ядра (release).

**Не проверено**: полный прогон `horizon-mixins plan/apply` и сборка игры с модом, а также игра на слух. Это
первое, что стоит сделать: `horizon-launcher native-build mixins/steam_audio --trust-native-code`.

## Чего нет (пока)

Окклюзия, отражения и реверберация по геометрии мира: в steam-audio-rust они есть (сцена, BVH, симуляторы),
но им нужен доступ к блокам мира из игрового потока и построение сцены из чанков — следующий шаг.

## Лицензия

Apache-2.0, как steam-audio-rust и Steam Audio. Steam Audio — © Valve Corporation; steam-audio-rust — ©
razed.production (см. `mixins/steam_audio/engine/NOTICE`). Мод не связан с Valve и не одобрен ею.
