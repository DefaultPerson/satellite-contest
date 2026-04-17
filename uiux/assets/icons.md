# Icon reference

Все иконки — из набора [Lucide](https://lucide.dev) (кроме `ton`, которая нарисована вручную). Исходники — в `assets/svgs` (текстовый файл с блоками SVG). Ниже таблица: что доступно и где планируется использовать.

| Name | Lucide id | Планируемое применение |
|---|---|---|
| `ton` | (custom) | Единица цены: «до 12 ◇» у покупки/нотификации |
| `dollar-sign` | `dollar-sign` | Маркер «цена покупки» в компактной карточке |
| `bell` | `bell` | Маркер «нотификации» — строка в карточке и в модалке |
| `bell-off` | — (добавить) | Состояние «нотификации выключены» |
| `arrow-down-up` | `arrow-down-up` | Сортировка в toolbar |
| `check` | `check` | Маркер «выбрано» в мультивыборе, success-бейджи |
| `chevron-down` | `chevron-down` | Фильтр-чипы «Коллекция ∨», «Фон ∨» |
| `plus` | `plus` | Кнопка «Добавить подписку», «пополнить баланс» |
| `cog` | `cog` | Настройки в шапке |
| `trophy` | `trophy` | Tab: Лидеры |
| `file-text` | `file-text` | Tab: Подписки |
| `search` | `search` | Tab: Поиск |
| `chart-column-increasing` | `chart-column-increasing` | Tab: Статистика |
| `user` | `user` | Tab: Профиль |
| `text` | `text` | Toolbar: переключатель «list» |
| `grid-3x3` | `grid-3x3` | Toolbar: переключатель «grid» |
| `x` | `x` | Закрыть модалку, закрыть webapp |
| `circle-check-big` | `circle-check-big` | Состояние «куплено / готово» |
| `circle` | `circle` | Пустой чекбокс в мультивыборе |
| `calendar-arrow-down` | `calendar-arrow-down` | Сортировка «по дате (свежие)» |
| `calendar-arrow-up` | `calendar-arrow-up` | Сортировка «по дате (старые)» |
| `arrow-down-0-1` | `arrow-down-0-1` | Сортировка «по количеству» |
| `arrow-up-1-0` | `arrow-up-1-0` | Сортировка «по количеству» |

## Нужно нарисовать/найти дополнительно (для ТЗ)

| Name | Назначение |
|---|---|
| `pencil` (lucide) | «Изменить» в `⋯` меню |
| `trash-2` (lucide) | «Удалить» в `⋯` меню |
| `copy` (lucide) | «Дублировать» в `⋯` меню |
| `more-vertical` (lucide) | `⋯` кнопка контекстного меню |
| `info` (lucide) | Кнопка информации в углу карточки |
| `bell-off` (lucide) | Нотификации выключены |
| `shopping-cart-off` (custom) | Покупка выключена |
| `alert-circle` (lucide) | Настойчивая индикация «куплено, 0 шт» |
| Логотипы маркетов | Portals, Fragment, MRKT, Tonnel, GetGems — заглушки потом |

## Цветовые состояния для иконок

- **Normal**: `var(--accent)` = `#f0f0f0`
- **Muted**: `var(--secondary-description)` = `#f0f0f04d`
- **Accent/active**: `var(--primary)` = `#5535f3`
- **Success**: `var(--color-success)` = `#79dd3b`
- **Destructive**: `var(--destructive)` = `#fc4a2f`
