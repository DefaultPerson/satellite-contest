# Satellite Contest

Проект для конкурса с четырьмя задачами: сортировка цветов, группировка коллекций подарков Telegram, анализ алгоритма распределения призов и UX-редизайн карточки подписки.

## Структура проекта

```
satellite-contest/
├── color-sorting/          # Задача 1: Сортировка цветов
│   ├── scripts/            # Python-скрипты с разными алгоритмами
│   ├── input/              # Входное изображение (80 цветов)
│   ├── output/             # Сгенерированные результаты
│   └── task.md             # Описание задачи
│
├── gift-collections/       # Задача 2: Группировка подарков
│   ├── scripts/            # Скрипты визуализации
│   ├── data/               # JSON с группами
│   ├── icons/              # Иконки подарков (107 PNG)
│   ├── output/             # Сгенерированные изображения
│   ├── background.png      # Фон для визуализаций
│   └── task.md             # Описание задачи
│
├── algorithm/              # Задача 3: Анализ алгоритма призов
│   ├── prize_simulation.ipynb  # Monte Carlo симуляция
│   └── task.md             # Условие конкурса
│
├── uiux/                   # Задача 4: UX-редизайн карточки подписки Gift Satellite
│   ├── solution/final/     # Финальный HTML-прототип (интерактивный)
│   ├── assets/figma-ready/ # Ассеты для Figma (SVG, PNG, logo)
│   ├── brand.md, task.md   # Дизайн-токены + ТЗ
│   └── README.md           # Ссылки на Figma (design + prototype)
│
├── research/               # Результаты исследований
│   ├── contest-answer.md   # Ответ для конкурса
│   └── portals-prize-algorithm-analysis.md
│
└── pyproject.toml          # Зависимости проекта
```

## Установка

```bash
uv sync
```

## Задача 1: Сортировка цветов

Отсортировать 80 цветов плавно от чёрного к белому. Реализовано 12 различных алгоритмов:

| Скрипт | Алгоритм |
|--------|----------|
| `sort_colors.py` | Базовая сортировка по HSL/яркости |
| `sort_colors_gradient.py` | TSP (nearest neighbor) в LAB |
| `sort_colors_human.py` | CIEDE2000 + TSP + 2-opt/3-opt |
| `sort_colors_oklab.py` | OKLab + монотонная яркость + simulated annealing |
| `sort_colors_diagonal*.py` | Диагональный градиент (3 версии) |
| `sort_colors_monotonic.py` | Строго монотонная яркость |
| `sort_colors_final.py` | Финальная оптимизированная версия |

### Запуск

```bash
cd color-sorting/scripts
uv run sort_colors_final.py
```

## Задача 2: Группировка подарков Telegram

Группировка 107 подарков в 18 тематических категорий:
- Новый год, Хэллоуин, День Валентина, Пасха
- Еда, Люкс, Техника, Азия, Игрушки
- Магия, Вечеринка, Snoop Dogg, Спорт, Космос
- Удача, Головные уборы, Природа, Напитки

### Запуск

```bash
cd gift-collections/scripts

# Одно изображение со всеми группами
uv run generate_groups_image.py

# Три отдельных изображения
uv run generate_groups_split.py
```

## Задача 3: Анализ алгоритма распределения призов Portals

Математический анализ алгоритма раздачи призов в конкурсе Portals (TON).

**Проблема:** Топ-7 участников с 20.3% билетов получили только 1.07% призового фонда (279 TON из 26,000).

**Решение:** Monte Carlo симуляция доказывает, что такой результат возможен только при розыгрыше с дешёвых призов или в случайном порядке. Вариант "с дорогих" исключён (ожидаемый выигрыш ~4200 TON, разница в 15x).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DefaultPerson/satellite-contest/blob/main/algorithm/prize_simulation.ipynb)

### Файлы

- `algorithm/task.md` — условие конкурса
- `algorithm/prize_simulation.ipynb` — симуляция Monte Carlo
- `research/contest-answer.md` — готовый ответ для конкурса

## Задача 4: UX-редизайн карточки подписки Gift Satellite

Редизайн карточки подписки в Telegram Mini App `@GiftSatelliteBot`. 16 состояний из ТЗ: 0/1/N моделей, фонов, узоров, dim/alert state'ы, bulk-select, info-modal.

- **Figma — Design file:** https://www.figma.com/design/qVhIG8Sdt5j1XHoSpf2dhl/Untitled?node-id=0-1&p=f
- **Figma — Prototype (со scroll):** https://www.figma.com/proto/qVhIG8Sdt5j1XHoSpf2dhl/Untitled?node-id=22-251&p=f&scaling=min-zoom&content-scaling=fixed&page-id=0%3A1
- **HTML-прототип:** [`uiux/solution/final/index.html`](uiux/solution/final/index.html) — интерактивный с 9 карточками, conic-gradient для 25 фонов, реальными Lottie-моделями

### Файлы

- `uiux/brif.md` — условия от @m1stervlad
- `uiux/task.md` — техническое задание (поля, состояния, правила)
- `uiux/brand.md` — дизайн-токены Gift Satellite (палитра, Hauora, радиусы)
- `uiux/solution/final/index.html` — итоговый HTML-прототип
- `uiux/assets/figma-ready/` — drag-drop ассеты для Figma (SVG-иконки, market-логотипы, model PNG, logo)

## Цветовые пространства

- **RGB** — базовая модель
- **HSL** — Hue-Saturation-Lightness
- **LAB/CIELab** — перцептуально равномерное пространство
- **OKLab** — современное перцептуальное пространство (2020)
