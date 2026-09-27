# Размещение на Render

Ветка: moscow-render, репозиторий: Va-Ya/MT-hack26. Push выполняется только после
отдельного разрешения владельца. Изменения репозитория сокомандника не публикуются.

В аккаунте Render выберите New → Blueprint, подключите свой GitHub-репозиторий
и ветку moscow-render. render.yaml создаёт один Docker Web Service Free и генерирует
серверный INGEST_TOKEN. Автоматические деплои отключены. Не выбирайте платный тариф.

Если создаёте Web Service вручную: Docker, Dockerfile.render, Free, Frankfurt,
ветка moscow-render, health check /api/health. Добавьте сильный INGEST_TOKEN,
DEMO_ENABLED=1, DATA_DIR=/app/demo/data, WEATHER_ENABLED=false, NDTP_ENABLED=0,
DETECT_STOPS=0. Start command оставьте из образа, порт определяется PORT.

После деплоя проверьте /, /api/health, /api/docs, /documentation/ и запуск ▶.
Готовность: model_version=e44988f343f4, schedule_available=true, затем появляются
ТС и прогнозы. /api/health/live проверяет процесс, /api/health — загрузку модели.

Серверный токен не включается в JavaScript. Для операторских действий вводится
в интерфейсе и хранится в памяти вкладки. Публичный replay ограничен окном данных.
Демо использует одно общее состояние. Журналы и импортированные данные теряются
при перезапуске бесплатного инстанса. Модель и встроенные данные остаются в образе.

Ключи Яндекс/Google можно добавить позже как build arguments VITE_YANDEX_MAPS_API_KEY
и VITE_GOOGLE_MAPS_API_KEY. Ограничьте browser key доменом своего сайта и разрешёнными API.
Без ключа API сайт продолжает работать с OpenStreetMap и реальными зонами наблюдений.

Пакет для сдачи: URL системы + ссылка на README репозитория, docs/JURY.md,
/documentation/ и /api/docs, docs/PERFORMANCE.md. Адрес сайта подставляется после деплоя.
