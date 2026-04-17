# Gift Satellite — Дизайн-система (библия)

Этот документ — **жёсткие правила** для всех вариантов редизайна. Задача: любой пользователь открыв прототип должен сказать «это Gift Satellite, просто улучшенная версия», без визуального шока.

> Предыдущие радикально-эстетические эксперименты (Velvet Vault, Phosphor Arcade, Broadsheet Brutal, Terrarium Glow, Glyph Terminal) отвергнуты и лежат в `solution/variants/archive-*/`. Новые варианты ОБЯЗАНЫ сохранять Gift Satellite brand look.

## Палитра (ЗАМОРОЖЕНО — не менять)

```css
/* Surfaces */
--background:     #101010;  /* true dark, page bg */
--foreground:     #1b1a1d;  /* base card surface */
--secondary:      #25242b;  /* surface 2: inactive chips, inputs, dropdowns */
--tertiary:       #3a3941;  /* borders, dim elements, switch-off track */
--tertiary-bg:    #18171c;  /* tooltips, sheets, modals bg */
--navbar-bg:      #18171cbf;/* bottom tab bar (blurred) */
--overlays:       #10101080;/* modal backdrops */

/* Brand */
--primary:        #5535f3;  /* purple — buttons, active chips, switch ON, links */
--primary-soft:   #5535f326;/* 15% primary for subtle fills */

/* Text */
--text:           #f0f0f0;  /* primary text */
--text-secondary: #f0f0f080;/* 50% */
--text-dim:       #f0f0f04d;/* 30% */
--outlines:       #f0f0f026;/* 15% — subtle borders */

/* Semantic */
--destructive:    #fc4a2f;  /* errors, 0 шт insistent, delete */
--destructive-soft: #fc4a2f26;
--success:        #79dd3b;  /* on, active, positive */
--success-soft:   #79dd3b26;
--warning:        #f0b429;  /* notify partial, warnings */
--warning-soft:   #f0b42926;
```

**Запрещено**: золотой, неон cyan/magenta, acid yellow, coral/sage/peach, terminal green, газетная palette.

## Типографика (ЗАМОРОЖЕНО)

**Основной шрифт**: Hauora, подключается локально из `assets/fonts/`:

```css
@font-face{
  font-family:'Hauora';
  src:url('../../../assets/fonts/Hauora-Regular.woff2') format('woff2');
  font-weight:400;font-style:normal;font-display:swap;
}
@font-face{
  font-family:'Hauora';
  src:url('../../../assets/fonts/Hauora-Medium.woff2') format('woff2');
  font-weight:500;font-style:normal;font-display:swap;
}
@font-face{
  font-family:'Hauora';
  src:url('../../../assets/fonts/Hauora-SemiBold.woff2') format('woff2');
  font-weight:600;font-style:normal;font-display:swap;
}

:root{
  --font-sans:'Hauora',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',sans-serif;
}
```

Опционально для цен: **JetBrains Mono** через Google Fonts (только weight 400/500). Больше никаких шрифтов.

### Шкала размеров
- `10-11px` — микро-лейблы, UPPERCASE letter-spacing 0.08-0.1em
- `12px` — small meta (коллекция, узоры)
- `13-14px` — body (статус, вторичный текст)
- `15-16px` — primary (имя подписки)
- `18-22px` — numeric (цены)
- `20-24px` — page headers

## Примитивы (ЗАМОРОЖЕНО)

### Карточка подписки
```css
background: var(--foreground);   /* или var(--secondary) */
border-radius: 16px;
border: 1px solid var(--outlines);
padding: 12-14px;
```

### Кнопка primary
```css
background: var(--primary);
color: var(--text);
border-radius: 10px;
height: 40-44px;
padding: 0 14-16px;
font-weight: 500-600;
```

### Кнопка secondary
```css
background: var(--secondary);
color: var(--text);
border-radius: 10px;
/* остальное как primary */
```

### Chip фильтра
```css
border-radius: 999px;
padding: 8px 14px;
font-size: 13-14px;
/* inactive: */ background: var(--secondary); color: var(--text);
/* active:   */ background: var(--primary);   color: var(--text);
```

### Switch
```css
border-radius: 999px;
width: 36-44px; height: 20-24px;
/* off: */ background: var(--tertiary);
/* on:  */ background: var(--primary);
/* thumb: white circle, smooth 200ms */
```

### Icon button
```css
width: 32-36px; height: 32-36px;
border-radius: 8px;
/* hover: */ background: rgba(255,255,255,0.06);
```

### Bottom tab bar (ЗАФИКСИРОВАНО)
```css
background: var(--navbar-bg);
backdrop-filter: blur(12px);
height: 64px + safe-area;
```
Содержит 5 пунктов в порядке: **Лидеры / Подписки / Поиск / Статистика / Профиль**. Активен «Поиск» — он выделен фиолетовой pill кнопкой, остальные — иконка сверху + dim label.

### Header (ЗАФИКСИРОВАНО)
- Лого 28-32px (из `assets/logo.jpg`), radius 8
- Тон-баланс: «1.92 ◇» + маленький `+`
- Cog-иконка настроек справа
- Высота 48-56px + safe-area top 40px

### Toolbar фильтров (ЗАФИКСИРОВАНО)
- Row chips: Коллекция ∨ / Фон ∨ / Поиск...
- Строка сортировки «Сначала новые» ↑↓
- Density toggle (list/grid) — опционально
- Счётчик «Всего 45 слотов | Создано 45 ордеров»
- CTA row: «Слот за 1 ◇» (primary) / «Добавить подписку» (secondary)

## Радиусы (единая шкала)
- 4-6px: бейджи, цветные фон-дотсы, маркеты
- 8px: icon-button, chip inner, мелкое
- 10px: кнопки
- 12-14px: модалки, sheets
- 16-18px: карточки
- 999px: pills, switches

## Space scale
4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32

## Иконки
Lucide style. Inline SVG. Stroke width 1.8-2px, rounded-linecap, rounded-linejoin. Размеры:
- 14-16px внутри badges
- 18-20px в карточке
- 20-22px в toolbar
- 24px в bottom-tab

## Motion
- Transitions: `200ms cubic-bezier(0.4, 0, 0.2, 1)` (standard easing)
- Switch toggle: 220ms
- Modal slide-in: 280ms ease-out
- НЕ использовать: spring overshoot, CRT flicker, snap-hard-cut, typewriter reveal. Только плавные стандартные transitions.

## Что РАЗРЕШЕНО менять между вариантами
- IA карточки (что где внутри)
- Density (размер карточки, 2×3 / 2×2 / 2×4 / 1×N)
- Размер превью, длина цепочки моделей, размер маркетов
- Визуальная иерархия (что крупнее/мельче — в рамках шкалы)
- Акценты состояний (badge position, top bar, dim, etc)

## Что менять НЕЛЬЗЯ
- Палитру (только токены из этого файла)
- Шрифт (только Hauora + системный fallback)
- Радиусы примитивов (chip pill, button 10px, card 16px, switch pill)
- Структуру header / toolbar / bottom-tab
- Иконки (только Lucide стиль)
- Base motion (только плавные transitions)

## Ассеты (относительно `solution/variants/vN-*/index.html`)
- Логотип: `../../../assets/logo.jpg`
- Шрифты: `../../../assets/fonts/Hauora-{Regular,Medium,SemiBold}.woff2`
- Превью подарков: `../../../assets/originals/{gift_id}/Original.png`
- Реальные цвета фонов: `../../../assets/data/backdrops.json`
- Имена коллекций: `../../../assets/data/id-to-name.json`

## Пример gift ID → name (для промптов)
- Ice Cream `5900177027566142759`
- Light Sword `5897581235231785485`
- Plush Pepe `5936013938331222567`
- Jolly Chimp `6005880141270483700`
- Jack-in-the-Box `6005659564635063386`
- Desk Calendar `5782988952268964995`
- Homemade Cake `5783075783622787539`
- Victory Medal `5830340739074097859`
- Heart Locket `5868455043362980631`
- Durov's Cap `5915521180483191380`
- Lol Pop `5170594532177215681`
- Sakura Flower `5167939598143193218`

## Пример backdrops (для реалистичности)
Из `backdrops.json` — Black `#363738`, Electric Purple `#ca70c6`, Lavender `#b789e4`, Cyberpunk `#858ff3`, Electric Indigo `#a980f3`, Neon Blue `#7596f9`, Navy Blue, Sapphire, Sky Blue, Pacific Cyan, Emerald, Mint Green, Pure Gold, Amber, Coral Red, Strawberry. Используй минимум 6 разных в демо-карточках.

## Правила для агентов
- Прочитать этот файл **первым делом**
- Следовать brand rules НЕ отклоняясь
- НЕ делать git commit/add без явной инструкции
- НЕ трогать архивные папки `archive-*`
- Писать в **назначенную** папку `v{N}-{slug}`, не в другое место
- Возвращать отчёт ≤80 слов
