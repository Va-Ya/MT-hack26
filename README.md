# Предиктор графика транспорта Москвы

Три модуля: ML-ядро ml/, backend FastAPI backend/, BI React dashboard/.
Backend строит признаки по отсечке T и вызывает ансамбль CatBoost.
Сервис не читает ответы из submission.csv: получает новые прогнозы из модели.

## Быстрый запуск

```powershell
docker compose up --build -d
```

Демо-данные включены. Дашборд http://localhost:8080, Swagger http://localhost:8000/docs,
метрики http://localhost:8000/metrics. Compose автоматически запускает replay.
Для полного датасета подключите data/raw и задайте DATA_DIR=/app/data/raw.

## Хостинг / единый образ

Для размещения без карты подготовлен [Railway](docs/RAILWAY.md):
пробный период до 30 дней или расходования $5. Конфигурация — railway.json.
Render также поддерживается, но проверенный аккаунт запросил привязку карты.

```powershell
docker build -f Dockerfile.render -t mt-hack26 .
docker run --rm -p 8000:8000 -e INGEST_TOKEN=YOUR_PRIVATE_TOKEN mt-hack26
```

Сайт http://localhost:8000, поток запускается автоматически; пауза и перемотка под картой. API /api/docs,
документация /documentation/. Модель работает на CPU в одном процессе.

## Материалы для жюри

- [Подача потока, NDTP, прогнозы, алерты](docs/JURY.md).
- [Размещение на Render](docs/RENDER.md).
- [Размещение на Railway без карты](docs/RAILWAY.md).
- [Производительность и дополнительные функции](docs/PERFORMANCE.md).
- [Сгенерированный PyDoc](docs/site/index.html).
- Swagger: /api/docs на сайте, :8000/docs в локальном Compose.

## ML и ограничения

Модель e44988f343f4, causal-v3, три CatBoost-модели, без переобучения.
Скор команды на платформе 0.93301; хеши победивших моделей сохранены.
История ограничена 20 минутами, event/receive/GPS должны быть доступны к T.
Горизонт (10,15] минут до планового события; фактическое упреждение показывается отдельно.
Факты не передаются в признаки раньше их доступности. Индекс риска не вероятность.

Демо — оригинальный тестовый поток, 13 ТС, весь доступный период с предысторией.
Границы воспроизведения берутся из датасета; даты не фиксируются в коде.
Без ключа API используется OpenStreetMap; координатная схема доступна как резервный режим. Внешние факторы не включены в модель.
Живой официальный ЦОДД endpoint не подтверждён. What-if — расчёт по предположениям.
Несколько uvicorn workers не поддерживаются: состояние replay в памяти.
На Free Render возможен холодный запуск и сброс состояния после перезапуска.

## Проверки

```powershell
python -m pip install -r requirements-lock.txt
python -m pytest -q
npm --prefix dashboard ci
npm --prefix dashboard run build
python -m ml.prepare
python scripts/benchmark.py
python scripts/build-docs.py
python scripts/publish-docs.py
```

Тест паритета на полном датасете требует data/raw. Статус контейнерных испытаний
и официального эмулятора указан в отчёте производительности.
