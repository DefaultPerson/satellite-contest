# Figma assets pack — drag-drop в один заход

Эта папка содержит все ассеты, нужные для доведения Figma-макета `Gift Satellite — Design System` до визуального паритета с `uiux/solution/final/index.html`.

## Быстрый старт

1. Открой Figma Desktop и нужный файл
2. **Drag-drop** эту папку (или отдельные файлы) прямо на холст — Figma создаст:
   - SVG → **Vector-нод** (можно перекрашивать через fill, ресайзить без потери качества)
   - PNG/JPG → **Rectangle с image fill**
3. Расклади по нужным местам через панель слоёв или по карте ниже

Tip: перекинуть сразу всю папку — Figma создаст отдельный фрейм "library" со всеми ассетами, из которого удобно copy-paste в целевые места.

---

## Карта использования

### `brand/`
| Файл | Где |
|---|---|
| `logo.jpg` | `Phone — Main → tg-bar → tg-logo` (28×28 radius 8) и `Phone — Main → app-head → brand → brand-logo` (32×32 radius 8). Заменить сиреневые placeholder-rect'ы. |

### `markets/` (5 маркетплейсов)
| Файл | Где | Размер |
|---|---|---|
| `portals.png` | market chip в buy/notify группах карточек | 12×12 circle radius 999 |
| `fragment.svg` | market chip | 12×12 |
| `mrkt.jpeg`   | market chip | 12×12 |
| `tonnel.png`  | market chip | 12×12 |
| `getgems.png` | market chip | 12×12 |

В Figma-макете заменить Unicode `●` в `buy-markets` / `notify-markets` text-node'ах. Карточка 1 (Victory): portals + fragment + mrkt + tonnel + getgems. Карточка 2 (Jack): portals + fragment. См. CARDS массив в `uiux/solution/final/index.html:580`.

### `models/` (9 Lottie→PNG моделей)
| Файл | Карточка |
|---|---|
| `victory-medal_aegis.png` | card-1 (Victory Medal, 1 модель) |
| `jack-in-the-box_aliens.png` | card-2 (Jack-in-the-Box · Aliens) |
| `light-sword_absinthe.png` | card-4 chain[0] |
| `light-sword_andromeda.png` | card-4 chain[1] |
| `light-sword_bifrost.png` | card-4 chain[2] |
| `ice-cream_apple-dip.png` | (не используется в 6 картах, для запаса) |
| `homemade-cake_angel-cake.png` | card-5 (Homemade Cake · sale) |
| `jolly-chimp_abubu.png` | card-6 chain[0] |
| `jolly-chimp_alarm-ape.png` | card-6 chain[1] |
| `victory-medal_brewmaster.png` | резерв |

Карточки 3 (Любой подарок), 5 (Plush Pepe · 12 моделей) и 8 (Desk Calendar · 0 моделей) показывают **collection image** — PNG лежат в `collections/`:

| Файл | Карточка | Оригинал |
|---|---|---|
| `collections/any-gift_5170594532177215681.png` | card-3 «Все коллекции» | ID 5170594532177215681 |
| `collections/plush-pepe_5936013938331222567.png` | card-5 Plush Pepe (12 моделей → иконка) | ID 5936013938331222567 |
| `collections/desk-calendar_5782988952268964995.png` | card-8 Desk Calendar (0 моделей → иконка) | ID 5782988952268964995 |

Размер в preview-wrap: 72×72 centered. Карточка 6 (Jolly Chimp rainbow) использует chain из 2 lottie моделей, а не collection image.

### `icons/` (29 Lucide SVG)

#### Header + toolbar
| Файл | Где в Figma |
|---|---|
| `x.svg` | `tg-bar → close-x` (заменить `✕`) |
| `more-horizontal.svg` | `tg-bar → tg-more` (заменить `⋯`) |
| `ton-diamond.svg` | `balance → icon-ton` (уже заменён через MCP; также `cta-primary` и 6 `price-big` в карточках) |
| `plus.svg` | `balance → plus-chip` (заменить text `+`) |
| `settings.svg` | `app-head → cog-btn → cog-ic` (заменить `⚙`) |
| `chevron-down.svg` | 2 chip'а toolbar `Коллекция ∨` / `Фон ∨` |
| `search.svg` | chip `Поиск...` и tab-bar Поиск |
| `sort.svg` | `sort-label` (заменить `↓↑`) |
| `check-circle.svg` | `selection-btn` (заменить `☐`) |
| `list.svg` | `density-list → d-list-ic` (заменить `☰`) |
| `layout-grid.svg` | `density-grid → d-grid-ic` (заменить `▦`) |

#### Card meta
| Файл | В каждой карточке |
|---|---|
| `package.svg` | `meta-collection` — заменить `📦` |
| `target.svg` | `meta-models` — заменить `⊙` |
| `image.svg` | `meta-backdrops` — заменить `🎨` |
| `grid.svg` | `meta-pattern` — заменить `▦` |
| `circle-info.svg` | `c-head → info-btn` — заменить `ⓘ` |
| `more-vertical.svg` | `c-head → more-btn` — заменить `⋮` |
| `shopping-bag.svg` | `buy-markets` — заменить `🛒` |
| `bell.svg` | `notify-markets` (on) — заменить `🔔` |
| `bell-off.svg` | `notify-markets` когда `notify.on=false` (card-8 Desk Calendar) |
| `alert-triangle.svg` | badge «0 шт» поверх card-2 (Jack alert) |

#### Tab bar — используй файлы из `icons/tab-bar/` (точные HTML-пути + inline stroke color)

| Файл | Таб | Stroke |
|---|---|---|
| `tab-bar/tab-leaders.svg` | Лидеры | `#f0f0f0` 50% (text-sec) |
| `tab-bar/tab-subs.svg` | Подписки (current, primary-soft pill) — **badge-check** (verified circle) | `#f0f0f0` 100% (text) |
| `tab-bar/tab-search.svg` | Поиск (primary pill + glow) | `#ffffff` 100% |
| `tab-bar/tab-stats.svg` | Статистика — outlined container с 3 столбцами | `#f0f0f0` 50% (text-sec) |
| `tab-bar/tab-profile.svg` | Профиль | `#f0f0f0` 50% (text-sec) |

Старые `trophy.svg` / `bookmark.svg` / `bar-chart.svg` в root `icons/` использовали generic Lucide-варианты. Папка `tab-bar/` содержит **точные пути из HTML final** (stroke-width 1.8, inline цвета) — рекомендуется использовать её для tab-bar и общих tab icons в других местах.

#### Modal / Bulk (когда будешь строить эти артборды)
| Файл | Где |
|---|---|
| `check.svg` | bulk-mode checkbox в выбранной карточке |
| `pencil.svg` | info-modal footer «Изменить» CTA |
| `trash.svg` | info-modal footer danger button + bulk-bar «Удалить» |
| `power.svg` | bulk-bar «Вкл» / «Выкл» buttons |

---

## Как перекрасить SVG в Figma

Все иконки нарисованы с `stroke="currentColor"` и шириной обводки 2 (Lucide-стандарт). В Figma:

1. Выдели импортированный vector-нод
2. В панели справа: **Stroke** → задай цвет (обычно `color/text` = `#f0f0f0` или `color/text-secondary` = `#f0f0f080`)
3. **Fill** обычно пустой (иконки line-art)
4. Для кнопок с primary фоном (tab-bar Поиск) stroke = `color/text` white

Диамонд `ton-diamond.svg` и `plus.svg` также работают по схеме stroke + currentColor.

## Связь с токенами

Все stroke-цвета привязываются к уже существующим Variables в коллекции «Gift Satellite»:
- Лейаут и рамки → `color/text-secondary` (50% opacity)
- Активные иконки (tab pill) → `color/text` (100%)
- Destructive (alert-triangle) → `color/destructive`

Через `Right panel → Stroke → Variables` назначаешь токен.
