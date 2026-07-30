# Граф-память brain в hermes: правила обновления и ограничения

Документ для того, кто обслуживает hermes-agent на junior (192.168.1.43). Описывает
локальные правки, которых НЕТ в upstream `NousResearch/hermes-agent`, и что с ними
делать при обновлении. Проверено 2026-07-28 на hermes `1d76a1518`. Два обновления подряд (26.07 — 2970 файлов,
28.07 — 318 коммитов) прошли чисто: патчи уцелели, чужая работа не откачена.

## Зачем это существует

У ботов две памяти, работающие **одновременно**:

- **mem0** — векторная, ищет по смыслу («что я говорил про сон?»).
- **brain** — граф знаний с типизированными рёбрами `субъект --предикат--> объект`
  (`tilapia --costs--> 359 uah per kg`). Отвечает на вопросы про **связи**, которых
  вектор не понимает: кто с кем, что откуда, через что связано.

brain работает не как MCP-инструмент, а как `MemoryProvider`, встроенный в цикл агента:
`prefetch` **принудительно** подмешивает связанные факты в контекст каждого хода,
`sync_turn` **принудительно** извлекает триплеты из хода через LLM и пишет их в граф.
Это ключевое отличие от MCP: боту не нужно «захотеть» воспользоваться памятью.

Хранилище — AlloyDB Omni на порту **5433**, отдельная БД на бота: `paul_brain`,
`fury_brain`, `bibi_brain`. CLI и доступ к БД — общие с nanobot-ботами:
`/home/dietpi/clawd/brain/cli.py` (пароль брать из `cli.DB_CONFIG`, не хардкодить).

## Инвентарь локальных правок

### 1. Три патча ядра — «multi-provider»

Upstream позволяет только ОДИН внешний провайдер памяти. Патчи снимают это ограничение,
чтобы mem0 и brain работали вместе. Каждый помечен в коде маркером `FORK PATCH (multi-provider)`.

| Файл | Что сделано |
|---|---|
| `agent/memory_manager.py` (~384) | В `add_provider` убран guard `self._has_external`. `MemoryManager` и так разветвляется по `self._providers` во всех местах (`prefetch_all`/`sync_all`/роутинг инструментов), guard давал только простоту конфига. Запрет на **повторную** регистрацию того же провайдера сохранён — иначе двойная запись. |
| `agent/agent_init.py` (~1400) | `memory.provider` читается как список: строка `"mem0, brain"` или YAML-список. Регистрируются все доступные. |
| `plugins/memory/__init__.py` (~360) | `_get_active_memory_provider()` при списке берёт первое имя (нужно только для сканирования CLI-подкоманд). |

Проверка наличия: `grep -rn "FORK PATCH" agent/*.py plugins/memory/__init__.py` → **ровно 3 попадания**.

### 2. Плагин brain — новый файл, которого нет в upstream

`plugins/memory/brain/__init__.py` (~27 КБ) + `README.md`. Это **untracked** для git.
Реализует `MemoryProvider`: `prefetch` (фоновый кэш, чтобы медленный бэкенд не тормозил
ход), `sync_turn` (фоновый поток → извлечение триплетов LLM → запись), `get_tool_schemas`
(инструмент `brain_graph`: `recall/neighbors/path/god_nodes/communities/entity/search/store/delete/update`),
`system_prompt_block`.

### 3. Правки общего brain CLI

`/home/dietpi/clawd/brain/cli.py` — вне hermes, но от него зависит:

- сетевой эмбеддер **mistral-embed (1024 измерения)** через OmniRoute для БД из
  `_OMNIROUTE_BRAINS` (строка ~48); ключ из env `OMNIROUTE_API_KEY` или файла
  `/home/dietpi/clawd/brain/.omniroute_key` (chmod 600, в `.gitignore`);
- неинтерактивные `delete`/`update` (флаги `-y/--force`, `--id`, `--content`);
  интерактивная версия вынесена в `_cmd_update_interactive`;
- `--force` удаляет **только при точном совпадении** `content`, иначе печатает id-кандидатов.

### 3b. Хранилище mem0 — СЕРВЕРНЫЙ Qdrant, не встроенный (с 27.07.2026)

`mem0.json` каждого профиля указывает на `"url": "http://127.0.0.1:6333"` — это
контейнер `qdrant` (bind-том `/mnt/data_disk/qdrant`, `restart: unless-stopped`),
поднятый владельцем сервера ещё в мае для индексации заметок и сессий
(`/home/dietpi/clawd/scripts/memory-index.sh`, `sessions-index.sh`). Коллекции
разделены по именам: наши — `paul_mem0` / `fury_mem0` / `bibi_mem0` (1024 измерения),
чужие — `memory` / `sessions` (768). **Не удалять чужие.**

Почему не встроенный режим (`"path": …`): встроенный Qdrant — однопользовательская
файловая база. Второй клиент в том же процессе получает
`Storage folder … is already accessed by another instance`, провайдер запоминает
ошибку в `_init_error` и **на всю сессию** отдаёт её вместо работы — все `mem0_add` /
`mem0_search` / `mem0_update` мертвы до перезапуска, молча, без единой ошибки в чате.
Ловилось 14.07, 17.07 и 27.07.

Старые встроенные каталоги (`~/.hermes/mem0_qdrant`, `…/profiles/fury/mem0_qdrant`)
намеренно НЕ удалены — это путь отката: вернуть `"path"` вместо `"url"` в `mem0.json`.
Бэкапы конфигов: `mem0.json.bak-qdrantserver-*`.

### 4. Конфиги (по профилю)

```yaml
memory:
  provider: mem0, brain        # ОБА, через запятую
plugins:
  hermes-brain:
    brain_db: paul_brain       # fury_brain / bibi_brain
    cli_dir: /home/dietpi/clawd/brain
    auto_store: true
    extract_model: minimax/MiniMax-M3
    recall_limit: 8
```

## Правила обновления hermes

**Шаг 1. Сохранить патчи ДО обновления.**

```bash
cd /home/dietpi/.hermes/hermes-agent
git diff > ~/hermes-forkpatches-$(date +%F).patch          # все правки отслеживаемых файлов
tar czf ~/hermes-brain-plugin-$(date +%F).tgz plugins/memory/brain
```

В `git diff` попадут не только 3 патча памяти — там есть и другие локальные правки
(`agent/auxiliary_client.py`, `agent/transports/chat_completions.py`, `cron/scheduler.py`).
Сохраняются все, разбираться по каждому отдельно.

**Шаг 2. Обновить.** Проверенный порядок (28.07.2026, 318 коммитов, без конфликтов):

```bash
git stash push -u -m "fork-patches-preupdate-<дата>" -- agent cron plugins/memory/__init__.py
cd /home/dietpi/.hermes && hermes update --yes     # официальный апдейтер: git + зависимости + venv + WebUI
cd hermes-agent && git stash pop
```

`git stash push` с явным списком путей НЕ трогает untracked-плагин `brain` — он остаётся
на месте, и это важно: в стандартный `-u` он бы попал целиком. Если `stash pop` даст
конфликт — применять правки по смыслу, цель каждой описана в инвентаре выше; резервный
путь `git apply --3way ~/hermes-forkpatches-<дата>.patch`.

При конфликте — применять правки по смыслу, а не по строкам: цель патчей описана в
инвентаре выше. Для `add_provider` смысл один: **не отклонять второй внешний провайдер**.

**Шаг 3. Проверить, что всё на месте** (все 4 проверки обязательны):

```bash
cd /home/dietpi/.hermes/hermes-agent
grep -rc "FORK PATCH" agent/agent_init.py agent/memory_manager.py plugins/memory/__init__.py
ls -l plugins/memory/brain/__init__.py
./venv/bin/python -c "import py_compile;py_compile.compile('plugins/memory/brain/__init__.py',doraise=True);print('ok')"
grep -n "provider:" ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml | grep -i memory -A0
```

**Шаг 4. Прогнать тест качества извлечения** — `/home/dietpi/clawd/brain/test_prompt_v4.py`:

```bash
cd /home/dietpi/.hermes && ./hermes-agent/venv/bin/python /home/dietpi/clawd/brain/test_prompt_v4.py
```

Ожидается «промпт v4 работает»: два кейса-провала дают 0 триплетов, контрольный — ≥5 верных.

Тест читает из живого файла плагина и `_EXTRACT_PROMPT`, и код-фильтр
`unsupported_precise_values`, применяя фильтр к ответу модели ровно так же, как это
делает `sync_turn`; креды берёт из `model.base_url` / `model.api_key` в
`~/.hermes/config.yaml`. Это единственный способ убедиться, что защита не потерялась
при мерже. Если падает первый кейс («ПРОСОЧИЛОСЬ»), значит потерян код-фильтр либо
раздел `WHOSE FACT IS IT` — восстановить из бэкапов `.bak-evidence-*` / `.bak-promptv4-*`.

**Шаг 5. Рестарт — обязательно с проверкой `HERMES_HOME`.**

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
for u in hermes-gateway hermes-gateway-fury hermes-gateway-bibi; do
  grep HERMES_HOME ~/.config/systemd/user/$u.service; done
```

Должно быть: `hermes-gateway` → `/home/dietpi/.hermes`, `-fury` → `.../profiles/fury`,
`-bibi` → `.../profiles/bibi`. 8 мая 2026 из-за подмены `HERMES_HOME` в unit-файле BiBi
встал на место Paul. Проверять КАЖДЫЙ раз перед рестартом.

**Шаг 6. Убедиться, что провайдеры поднялись:** в `${HERMES_HOME}/logs/gateway.log`
не должно быть `Rejected memory provider` — эта строка означает, что патч
`memory_manager.py` потерян и brain отключён.

**Шаг 7. Проверить заглушённые скилы:** `hermes skills list-modified` (и то же с
`--profile fury` / `--profile bibi`). Локальные скилы в `${HERMES_HOME}/skills`
перекрывают бандл-овые ПО ИМЕНИ, а `hermes update` намеренно не трогает то, что счёл
пользовательской правкой. Из-за этого пустой остаток каталога глушит рабочий скил
навсегда и молча: 28.07 так были отключены `ocr-and-documents` у всех трёх профилей —
локально лежал только `DESCRIPTION.md` (147 байт), без `SKILL.md` и `scripts`, и скил
вообще не появлялся в `hermes skills list`. Лечится `hermes skills reset <name>
--restore --yes` на каждом профиле, затем рестарт.
Отличать поломку от настоящей правки по `hermes skills diff <name>`: «only in stock:
SKILL.md» — это поломка; нормальный дифф со строками `+`/`-` — осознанное изменение
(так у BiBi в `google-workspace` добавлена ссылка на `references/email-via-cli.md`,
её трогать не надо).

## Ограничения — чего делать НЕЛЬЗЯ

1. **Не запускать `git clean -fd`/`-fdx` в `hermes-agent`.** Плагин brain — untracked,
   clean уничтожит его безвозвратно. По той же причине не делать `git reset --hard` и
   `git checkout .` без предварительного `git diff > файл`.
2. **Не возвращать guard одного внешнего провайдера** в `add_provider`. Если он вернётся,
   brain молча отключится: mem0 регистрируется первым, brain будет отклонён, а бот
   продолжит отвечать — деградация без ошибки.
3. **Не менять `memory.provider` на одно значение.** Должно остаться `mem0, brain`.
   Порядок важен: первое имя используется для сканирования CLI-подкоманд.
4. **Не убирать `-y` из вызова `store`** в `_store_triple`. Без него `cmd_store` вызывает
   `input()`, а в subprocess это `EOFError` → авто-запись памяти умрёт целиком.
5. **Не ослаблять `--force` у `delete`/`update` до похожести по вектору.** Раньше порог
   был 0.75 — кросс-языковые эмбеддинги дают 0.75+ на несвязанном тексте, и тест удалил
   живой факт. Только точное совпадение `content`; при промахе brain печатает id, дальше
   удалять через `--id`.
6. **Не менять модель эмбеддинга без миграции БД.** Колонки `entities.embedding` и
   `facts.embedding` — `vector(1024)` под mistral-embed. Другая модель = другая
   размерность = сломанный recall. У mistral-embed нельзя передавать параметр
   `dimensions`/`embedding_dims` — вернёт HTTP 422.
7. **Не ослаблять правила промпта извлечения** (`_EXTRACT_PROMPT`, раздел
   `WHOSE FACT IS IT`). Они закрывают конкретную аварию: бот записывал в память свои же
   галлюцинации при чтении фото чека — один чек оказался подписан тремя разными
   магазинами и валютами, — и свою саморефлексию («у меня системная ошибка»). Два принципа:
   факт должен переживать забвение диалога; своё чтение картинки — гипотеза, а не
   свидетельство, поэтому считанные с фото ярлыки (магазин, город, дата, валюта, номер
   документа) не пишутся, если пользователь не подтвердил их текстом.
   Формулировать **принципами, а не перечислением случаев**: MiniMax строго следует
   правилам и перечисление читает как исчерпывающий список — на предыдущей версии
   промпта он из-за этого вообще перестал извлекать покупки.
8. **Не убирать код-фильтр `unsupported_precise_values`** в плагине brain и его вызов
   в `sync_turn`. Правило: точная величина или идентификатор (цена, вес, номер
   документа, индекс) попадает в память, только если её написал пользователь; мелкие
   счётные числа не проверяются. Это НЕ дубль промптового правила: 27.07 та же модель
   с тем же промптом снова протащила выдуманные магазин, город и номер чека —
   промпт-защита зависит от послушности модели и от того, какой провайдер сегодня
   стоит за алиасом, а код — нет. Проверяется тестом (шаг 4).
9. **Не убирать защиту от самопетель** в `_store_triple` (`subject == object` →
   `"skipped self-loop"`). Правило есть и в промпте, но код-страховка не зависит от
   послушности модели.
10. **Не трогать `/home/dietpi/clawd/brain/cli.py` без оглядки на nanobot** — этот файл
   общий с ботами thufir/leto/wysd/creatio. `creatio_brain` намеренно остался на локальном
   Ollama `nomic-embed-text` (768), остальные — на mistral-embed (1024). Список в
   `_OMNIROUTE_BRAINS`; новую БД бота надо добавлять туда же.
11. **Не полагаться на форму ребра при чистке данных.** Ребро может быть идеально
    типизированным и при этом полностью выдуманным. Перед удалением смотреть
    `content` и сверять числа с подтверждёнными фактами. И наоборот: узлы вида
    `359 uah per kg`, `162.32 uah`, `shevchenko street 60, lviv` — это норма, так схема
    хранит значения, а не мусор.
12. **Не переводить mem0 обратно на встроенный Qdrant** (`"path"` вместо `"url"`) и не
    выключать контейнер `qdrant` — от него теперь зависит память всех трёх ботов.
    Проверка живости: `curl -s http://127.0.0.1:6333/collections` должен показывать
    `paul_mem0`, `fury_mem0`, `bibi_mem0` рядом с чужими `memory`, `sessions`.
13. **Перед массовой правкой данных делать бэкап таблиц:**
    `CREATE TABLE facts_bak_<ts> AS SELECT * FROM facts` (и то же для `entities`).
    Так уже восстанавливали ошибочно удалённый факт.

## Откат

- Плагин: `plugins/memory/brain/__init__.py.bak-*` (последние — `.bak-promptv4-20260726_205350`,
  `.bak-selfloop-20260726_205733`).
- Ядро: `agent/*.py.bak-multiprov2-*`, либо `git checkout -- <файл>` для возврата к upstream
  (это отключит brain — см. ограничение 2).
- Данные: таблицы `entities_bak_*` / `facts_bak_*` в каждой `*_brain` БД.
- Конфиги: `config.yaml.bak-*` в каждом профиле.
