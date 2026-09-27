# Развёртывание

Локальный Compose: `docker compose up --build -d`, dashboard http://localhost:8080. Исходные данные должны быть распакованы в `data/raw`; они не входят в Git. Модели включены в репозиторий.

## Внешний сервер

1. Склонируйте репозиторий на Linux-хост с Docker Compose, распакуйте архив через `scripts/unpack.py`.
2. Создайте `.env` из `.env.example`. Настройте Google browser key и сильный `INGEST_TOKEN`.
3. Выполните `docker compose up --build -d`.
4. Настройте HTTPS reverse proxy на localhost:8080. Порты backend не публикуйте напрямую.
5. Ограничьте доступ диспетчера: HTTP-аутентификация или VPN. Управление replay доступно через `/api/replay/control`; сервер проверяет `X-Ingest-Token`. Токен вводится через «Доступ к управлению» и не включён в сборку JS.
6. Ограничьте Google key доменом сайта и Maps JavaScript API. Он публичный по назначению, серверный INGEST_TOKEN — нет.

Публичного deployment пока нет. GitHub хранит исходники; GitHub Pages не запускает Python/CatBoost backend.

## Проверка

- `docker compose ps`: backend healthy, dashboard запущен.
- `/api/health`: ML ONLINE.
- В replay растёт число прогнозов, затем observed_outcomes.
- Play/Pause и перемотка управляют тем же worker, что запускает контейнер emulator.
- Google-тайлы требуют действующего ключа; их ошибка не должна блокировать replay/ML.
- `docker compose run --rm backend python -m pytest -q` проверяет ML и API внутри Linux-контейнера.

Не масштабируйте backend несколькими workers: replay и оперативная история находятся в памяти процесса. Для переноса на другой хост нужны исходные данные, models и конфигурация `.env`.
