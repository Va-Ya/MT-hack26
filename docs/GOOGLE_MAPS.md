# Google Maps: подключение

Frontend переведён на Google Maps JavaScript API. Надписи MVP убраны из боковой панели и footer. Тепловой слой рисуется Canvas через официальный OverlayView; устаревший google.maps.visualization.HeatmapLayer не используется. Яндекс/Leaflet/OSM как провайдеры удалены.

## Что нужно сделать владельцу проекта

1. В своём Google Cloud проекте подключить billing и включить **Maps JavaScript API**. Это действие с платёжным аккаунтом выполняется владельцем.
2. Создать browser API key. В Application restrictions выбрать Websites (HTTP referrers), в API restrictions оставить Maps JavaScript API. Для локального запуска разрешить `http://localhost:5173/*`, `http://127.0.0.1:5173/*`, а для Compose — соответствующие адреса с портом 8080. После публикации добавить свой HTTPS-домен. Не использовать unrestricted серверный ключ.
3. Вписать ключ в `.env` **в корне проекта**:

```dotenv
VITE_GOOGLE_MAPS_API_KEY=ваш_ключ
```

4. Перезапустить Vite. Он читает корневой `.env` через envDir. Для Docker выполнить `docker compose up --build`: ключ передаётся в frontend при сборке. Изменение ключа требует новой сборки production bundle.
5. Открыть карту и проверить авторизацию Google, появление heatmap, zoom и выбор проблемной зоны. Без ключа отображается явное сообщение о настройке; чужая карта под видом Google не показывается. API, модель и журнал продолжают работать.

Ключ Maps JavaScript API предназначен для браузера и присутствует в клиентском bundle; защиту дают ограничения по API и доменам. `.env` исключён из Git. Backend этот ключ не использует. Map ID, Geocoding/Places API и ключ Яндекса не нужны.

## Проверено / не проверено

Проверяются TypeScript, production build, отсутствие надписей MVP и корректный экран без ключа. Загрузку реальной карты Google, ограничения ключа и позиционирование overlay на работающей карте можно подтвердить только с действующим ключом. На момент разработки ключа в проекте нет.

## Официальные источники

- [Настройка Maps JavaScript API, billing и ключа](https://developers.google.com/maps/documentation/javascript/get-api-key)
- [Загрузка JavaScript API](https://developers.google.com/maps/documentation/javascript/load-maps-js-api)
- [OverlayView и пользовательские слои](https://developers.google.com/maps/documentation/javascript/customoverlays)
- [Отключение старого Heatmap Layer в мае 2026](https://developers.google.com/maps/deprecations)
