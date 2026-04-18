# Gift Satellite — Redesign карточки подписки

Конкурс **IT Monday** от Gift Satellite. UX-редизайн карточки подписки в главном меню Telegram Mini App `@GiftSatelliteBot`.

- **Дедлайн:** пятница, 17 апр 2026
- **Формат сдачи:** макет в Figma + HTML-прототип
- **Призы:** 1-е — 200 USDT + Lol Pop + возможный контракт; 2-е — 100 USDT

## Артефакты

| Что | Где |
|---|---|
| 🎨 **Figma — Design file** | https://www.figma.com/design/qVhIG8Sdt5j1XHoSpf2dhl/Untitled?node-id=0-1&p=f |
| ▶️ **Figma — Prototype** (со scroll) | https://www.figma.com/proto/qVhIG8Sdt5j1XHoSpf2dhl/Untitled?node-id=22-251&p=f&scaling=min-zoom&content-scaling=fixed&page-id=0%3A1 |
| 🧩 **HTML-прототип** | [`solution/final/index.html`](solution/final/index.html) — интерактивный, 9 карточек, conic-gradient 25 фонов, Lottie-модели |
| 🗂 **Solution hub** | [`solution/index.html`](solution/index.html) — индекс 5 финальных вариантов + архив |
| 🎁 **Ассеты для Figma** | [`assets/figma-ready/`](assets/figma-ready/) — SVG-иконки, 5 market-лого, 9 model PNG, collections PNG, brand logo |

## Структура

```
uiux/
├── README.md                  ← ты здесь
├── brif.md                    ← условия от @m1stervlad
├── task.md                    ← ТЗ: все 16 состояний карточки
├── brand.md                   ← дизайн-токены: палитра, Hauora, радиусы
├── plan.md                    ← рабочий план/концепция (исторический)
│
├── orig/                      ← оригинал UI — скрины + видео для референса
│
├── research/
│   ├── ui-audit.md            ← gap-анализ против ТЗ
│   ├── chat-feedback.md       ← фидбек из чата
│   └── dev/lottie-test.html   ← песочница для .tgs → PNG
│
├── assets/
│   ├── figma-ready/           ← ассеты для drag-drop в Figma ⭐
│   │   ├── icons/             ← 29 Lucide SVG (tab-bar + meta + UI)
│   │   ├── markets/           ← portals/fragment/mrkt/tonnel/getgems
│   │   ├── models/            ← 10 Lottie→PNG моделей
│   │   ├── collections/       ← 3 collection-иконки для >4 моделей
│   │   └── brand/logo.jpg     ← логотип приложения
│   │
│   ├── data/                  ← реальные данные: backdrops, patterns, id-to-name
│   ├── fonts/                 ← Hauora (Regular/Medium/SemiBold .woff2 + .ttf)
│   ├── market-logos/          ← исходники маркетов (SVG/PNG)
│   ├── models-png/            ← Lottie → PNG rendering
│   ├── colors.css             ← CSS с :root дизайн-токенами
│   └── *-tgs → symlinks       ← .tgs оригиналы из tg-gifts-autotrade-system
│
└── solution/
    ├── index.html             ← hub с 5 вариантами + архив
    ├── final/index.html       ← финальный прототип (выбран для сдачи)
    └── variants/              ← 5 финальных + 5 архивных экспериментов
```

## Бренд

- **Палитра:** `brand.md` / `assets/colors.css` → `:root { --primary: #5535f3; --bg: #101010; ... }`
- **Шрифт:** Hauora (Regular/Medium/SemiBold) с fallback на Roboto
- **Иконки:** [Lucide](https://lucide.dev) с `stroke="currentColor"`
- **Тема:** только тёмная

## Ключевые решения

- **Промпт-имя пользователя** = главный заголовок карточки (truncate ≤40 симв)
- **Превью фонов:** 1 фон → radial, 2-5 → horizontal bands, >5 → conic-gradient
- **Превью моделей:** 1-4 конкретных → Lottie, 0 или ≥5 → collection-иконка
- **Info** → bottom-sheet модалка с полным описанием подписки
- **Bulk-select** через кнопку в toolbar (long-tap блокируется iOS Telegram)
- **Маркеты:** Portals, Fragment, MRKT, Tonnel, GetGems (отображаем до 2 + «+N»)
- **Dim state** при `buy.on = false`; **alert state** при `count = 0`

Все 16 состояний из `task.md` покрыты 9 карточками прототипа.

## Как запускать

```bash
cd uiux
python -m http.server 8080
# http://localhost:8080/solution/          — hub с 5 вариантами
# http://localhost:8080/solution/final/    — финальный прототип
```
