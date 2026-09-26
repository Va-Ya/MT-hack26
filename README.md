# Предиктор задержек транспорта Москвы

CSV → единый причинный FeatureBuilder → CatBoost → StreamingPredictor → FastAPI → диспетчерский dashboard с Google Maps.

## Быстрый запуск

Требуются Docker Engine / Compose и исходный архив хакатона. Данные не входят в Git: это сохраняет компактный push. Обученные модели и submission включены.

```powershell
python scripts/unpack.py "C:\path\Предиктор задержек транспорта.zip"
Copy-Item .env.example .env
# В .env задайте VITE_GOOGLE_MAPS_API_KEY для карты.
docker compose up --build -d
```

Открыть **http://localhost:8080**, Swagger **http://localhost:8000/docs**. Сервисы: подготовка ML, backend, dashboard и запуск управляемого replay. Порты доступны только локально. Остановка: `docker compose down`.

Google Maps требует browser key, включённого Maps JavaScript API и billing. Ограничьте ключ доменами сайта и Maps JavaScript API. Инструкции: [GOOGLE_MAPS](docs/GOOGLE_MAPS.md). Без ключа доступны управление replay, показатели, hotspots и графики; фон карты показывает состояние «Нужен ключ». Реальная загрузка Google-тайлов требует пользовательского ключа.

Если задан `INGEST_TOKEN`, введите его в панели «Доступ к управлению». Токен хранится только в памяти вкладки. Для публикации в интернете используйте HTTPS и обязательную защиту управления, см. [DEPLOYMENT](docs/DEPLOYMENT.md).

## Dashboard features

- Плотное рабочее место: статусная строка, фильтры слева, большая карта, hotspots справа, подробности снизу.
- Плавная heatmap на базе реальных координат и ячеек около 650 м; Google Maps OverlayView, без удалённого Google HeatmapLayer.
- «Сейчас» / «Прогноз 10–15 минут»: модель предсказывает первое плановое событие в окне (10,15], отдельных моделей +10/+15 нет.
- Hotspots: индекс риска, текущее отклонение, прогноз, число ТС, горизонт и временной тренд.
- История зон: до 30 минут, снимки через 30 секунд времени источника. Тренд = риск сейчас минус риск 5 минут назад; при отсутствии прошлых наблюдений показывается недостаток истории.
- Объяснимые вклады в индекс риска и рекомендации по правилам. Это не SHAP и не автоматические команды.
- Replay: LIVE / REPLAY, настоящие Play/Pause, ×1/×10/×50, timeline и перемотка. Перемотка пересобирает состояние в порядке доступности событий и заканчивается паузой. Подготовка может занимать несколько секунд.
- Прогноз → факт: выбор ТС и конкретного выданного прогноза, ошибка, реальное упреждение до факта и график прогнозов одного целевого события.
- Поиск по tr_id, адресу и ID планового остановочного события. Это не идентификатор физической остановки.
- Фильтр риска, переключение heatmap/транспорта, группы ячеек на zoom 12–13 и ТС с zoom 14. API фильтрует координаты по bbox; отдельные ТС загружаются для области карты или выбранной зоны.
- Обновление каждые 2,5 секунды; сохранение последнего состояния при потере API; отдельные состояния модели и внешней карты.

В источнике нет идентификаторов маршрутов, рейсов, типов транспорта и геометрии маршрутов. Эти данные не выдумываются. Округа, пробки, дорожные события, погода и what-if не подключены. Необязательные слои не блокируют ML.

## Demo scenario · 1–2 минуты после подготовки

1. Откройте dashboard. Replay запускается автоматически в Docker; нажмите «Пауза».
2. На timeline выберите около **07:15**, нажмите «Перейти» и дождитесь подготовки.
3. Переключите «Сейчас» → «Прогноз 10–15 мин», выберите hotspot с самым высоким риском.
4. Покажите текущую задержку, прогноз, причины риска, транспорт и рекомендации.
5. Через поиск выберите **ТС 122048**, откройте «ТС · прогноз и факт» и закрепите один прогноз в списке «Выданный прогноз».
6. Нажмите «Старт», скорость ×50. Прогноз остаётся выбранным, факт раскрывается после события.
7. Либо перемотайте к **07:35** и дождитесь восстановления. Покажите факт, абсолютную ошибку и упреждение того же прогноза.
8. Вернитесь к зоне и покажите историю. Смена зоны может дать недостаток истории: это реальные перемещения ТС между ячейками.

Красная зона не гарантируется на любой секунде; индекс зависит от фактического состояния. Демо использует данные 6 января 2026 года, а не текущую Москву. Временная зона исходных меток не указана, используется шкала источника.

## Результат ML

Temporal MAE на поздних реальных объектах train: **89.16 с**, baseline `cur_dev_s`: **95.55 с**. Отдельный поздний test: **78.50 с** против **88.19 с**. Данных примерно за сутки; результат не доказывает качество на новых днях. Подробности: [ML_REPORT](docs/ML_REPORT.md), [DATA_MODEL](docs/DATA_MODEL.md).

- `models/best_model.cbm`: финальная модель только на labels_train.
- `models/temporal_model.cbm`: модель для временной оценки.
- `models/metadata.json`: версия, признаки и окно прогноза.
- `artifacts/submission.csv`: UTF-8, `sample_id;prediction`, 151 строка.
- `artifacts/experiments.csv`, `feature_importance.csv`, `error_analysis.csv`, `metrics.json`: результаты экспериментов.
- Runtime `artifacts/prediction_log.parquet`: последние 20000 прогнозов; архив предыдущего запуска в `artifacts/logs/`. Эти файлы исключены из Git.

Все признаки используют только телеметрию, доступную к T (event/receive/GPS ≤ T). История ограничена 20 минутами. Факт поступает отдельным событием; будущие факты не передаются в FeatureBuilder. Управляемый replay предварительно воспроизводит 20 минут перед началом демонстрационного окна 06:50–08:00.

## Локальная разработка

Python 3.12, Node 24:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
npm --prefix dashboard ci
.\scripts\start-local.ps1
```

Открыть http://127.0.0.1:5173 и нажать «Старт». Остановка локальных процессов: `scripts/stop-local.ps1`. На Linux запускайте `.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` и `npm --prefix dashboard run dev`.

Дополнительный CLI replay сохранён для интеграционной проверки. Сначала переключите dashboard в LIVE, чтобы два источника не писали одновременно:

```powershell
.\.venv\Scripts\python.exe -m emulator.replay --speed 50 --start "2026-01-06 06:50:00" --end "2026-01-06 08:00:00"
```

CLI не управляется кнопками managed replay. Для нового CLI-сеанса с более ранним временем переключите LIVE заново для сброса состояния.

## Проверки и воспроизведение обучения

```powershell
.\.venv\Scripts\python.exe -m pytest -q
npm --prefix dashboard run build
.\.venv\Scripts\python.exe -m ml.data_audit
.\.venv\Scripts\python.exe -m ml.build_dataset
.\.venv\Scripts\python.exe -m ml.train
.\.venv\Scripts\python.exe -m ml.make_submission
```

14 тестов: отсутствие утечки, паритет offline/online, отложенные факты, идемпотентность, ограничение истории, API, risk aggregation, временной тренд и causal seek. ML pipeline сохранён.

## API

Основные: `POST /telemetry`, `GET /vehicles?bbox=west,south,east,north&cell_id=...`, `GET /vehicles/{id}`, `GET /network/heatmap?mode=current|forecast&bbox=...&zoom=10&min_risk=20`, `GET /hotspots`, `GET /health`, `GET /metrics`, `GET /external/status`.

Добавлено:

- `GET /replay/state` — режим, статус, скорость, границы и курсор.
- `POST /replay/control` — `{action: play|pause|seek|speed|live, speed?: 1|10|50, timestamp?: ...}`.
- `GET /network/zones/{cell_id}/history?mode=forecast` — до 30 минут истории.
- `GET /search?q=...` — ТС и плановые события по ID/адресу.
- `GET /predictions?tr_id=...&target_stop_id=...&observed=true&limit=1000` — фильтры журнала.

Технические: `POST /replay/context`, `/replay/outcome`, `/replay/checkpoint`. Все POST проверяют `X-Ingest-Token`, если задан `INGEST_TOKEN`. Nginx разрешает из dashboard только POST управления replay, остальные POST остаются на прямом backend. Не передавайте будущие outcome-события в живую интеграцию.

## Структура и ограничения

`ml/` — аудит, признаки, обучение, inference и streaming; `backend/` — API, геоагрегация, история и replay controller; `emulator/` — CSV replay; `dashboard/src/` — React/Google Maps; `tests/` — проверки; `scripts/` — распаковка/локальный запуск; `docs/` — данные, ML, карта и deployment.

История и replay controller хранятся в памяти одного backend-процесса; не запускайте несколько uvicorn workers. После перезапуска состояние восстанавливается новым replay. Время обработки модели не включает сеть и запись Parquet. Replay MAE включает коррелированные прогнозы одного события и не заменяет temporal validation.
