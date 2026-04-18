# Research: HTML → Figma для `uiux/solution/final/index.html`

**Дата**: 2026-04-18
**Режим**: Technical Research
**Статус**: Final
**Исходник**: `uiux/solution/final/index.html` — 973 строки, mobile-прототип 393×852 (iPhone frame), dark theme, кастомные шрифты Hauora (`@font-face`), CSS-переменные, вложенный flex, Lottie-плеер, ~136 классов, ~54 SVG/img/Lottie.

## Executive Summary

В 2026 году для переноса HTML в **редактируемый** Figma-файл с минимальным ручным трудом реально работают три пути. Победитель — плагин **html.to.design** от divRIOTS: новая вкладка «Import code» (запущена в 2025) парсит HTML/CSS прямо из буфера, сохраняет Auto Layout из flex, резолвит CSS-переменные и даёт ~85–90 % редактируемых слоёв. Бесплатная альтернатива — **Builder.io Visual Copilot** (Auto Layout всё ещё в beta, придётся часок чистить). Готового Claude Code skill «HTML → Figma» не существует; уже стоящий MCP `figma-mcp-go` умеет писать в Figma (73 инструмента), но DOM-парсера у него нет — нужно писать Puppeteer-мост самому (1–2 дня).

## Рекомендация

**Primary**: `html.to.design` — вкладка «HTML» в плагине → вставить содержимое `index.html`. 🟢
**Fallback бесплатный**: Builder.io Visual Copilot Figma plugin. 🟡
**Power path без подписки**: Puppeteer (`getComputedStyle` + `getBoundingClientRect`) → JSON → `figma-mcp-go` (`create_frame` + `set_auto_layout` + `create_text` + `set_fills`). 🟡

## Сравнение инструментов

### html.to.design (divRIOTS) 🟢 — РЕКОМЕНДОВАНО
- **Цена**: Free 10 импортов / 30 дней; Pro ~$12–15/мес.
- **Fidelity**: 9/10. Редактируемые слои + Auto Layout из flex (direction, gap, padding, align/justify), фонты автоматом мапятся если установлены в Figma, CSS-переменные схлопываются в значения, есть mobile preset 390 px (ставим 393).
- **Как использовать для твоего кейса**:
  1. Установить шрифты `Hauora-Regular/Medium/SemiBold` в Figma desktop.
  2. Figma → Plugins → html.to.design → вкладка «HTML» → paste содержимого `index.html`.
  3. Либо `python -m http.server 8000` + Chrome extension html.to.design.
- **Минусы**: Lottie → static PNG (надо отдельно импортировать через LottieFiles-плагин), `@font-face` с URL-ссылкой на woff2 не сработает, нужны локальные шрифты.

### Builder.io Visual Copilot (Figma plugin) 🟡
- **Цена**: Free с кредитами.
- **Fidelity**: 7/10, сами заявляют 80–90 %.
- **Плюсы**: бесплатно, активно развивают, «Design with AI» для доводки.
- **Минусы**: Auto Layout всё ещё в beta — вложенный flex часто схлопывается в absolute; на Reddit в 2025 г. жалуются на «convoluted» UX.

### Figma Dev Mode MCP server (official) 🟡
- GA с 2025, в 2026 появился write-to-canvas + `generate_figma_design` на `https://mcp.figma.com/mcp`.
- Умеет снять live URL через Playwright и положить во frames.
- **Не подходит**: требует работающий браузер/localhost, не умеет raw `.html` файл; резолвит в raw frames, не в твои library-компоненты.

### figma-mcp-go (vkhanhqui) 🟢 уже установлен у тебя
- 603★, MIT, Go, активный. Bridge через Figma plugin — обходит лимиты REST API бесплатного плана.
- 73 инструмента write: `create_frame`, `set_auto_layout`, `create_text`, `set_fills`, `set_strokes`, `set_corner_radius`, `set_effects`, `create_variable_collection`, `bind_variable_to_node`, `import_image` и т.д.
- **Не имеет** HTML-импортера и open-source скрипта, который ходил бы по DOM и эмитил эти вызовы.
- **Для твоего кейса**: ~300–800 нод × 2–4 вызова = 1500–3000 последовательных tool calls. Долго, токенозатратно, легко словить edge-cases (grid, `::before`, `calc()`, absolute, gradients).
- **Правильное применение**: *после* html.to.design — довести руками (compose components, привязать CSS-переменные к Figma Variables, переименовать слои).

### Anima
- **Неправильное направление** (Figma → code). Исключить.

### html2design.com
- $12/мес, no free tier. Fidelity 8/10. Новее, меньше сообщество. Не тестировался широко.

### Claude Code skills / plugins
- Purpose-built «HTML → Figma» skill'а **нет** (апрель 2026).
- `markacianfrani/claude-code-figma` — примитив, только «создать прямоугольник».
- `frontend-design` (стоит у тебя) — генерирует HTML, не импортирует.
- `GLips/Figma-Context-MCP` — **read-only**, не запишет.
- В awesome-claude-skills и awesome-claude-plugins — пусто по теме.

## Под твой HTML (specific)

| Ограничение | html.to.design | Builder.io | figma-mcp-go + bridge |
|---|---|---|---|
| Frame 393×852 | preset 390 → 393 | вручную | тривиально |
| Dark theme CSS | ок | ок | ок |
| `@font-face` Hauora | **ставить в Figma заранее** | то же | то же |
| CSS `var(--*)` | резолвятся в значения | резолвятся | `getComputedStyle` выдаст значения |
| Вложенный flex → Auto Layout | лучший в классе | beta, частично | ты контролируешь 100 % |
| Lottie-плеер | → static PNG, вручную | → static PNG | импорт как отдельный asset |
| SVG-иконки (54 шт) | ок | ок | через `import_image` или inline-path |

## План действий без головной боли

1. Установить Hauora в Figma desktop (drop три woff2 из `uiux/assets/fonts/` — Figma подхватит через FontBook/системные шрифты).
2. Figma → Plugins → install `html.to.design` → бесплатный тариф.
3. Открыть плагин → вкладка «Import code / HTML» → вставить содержимое `index.html` → импорт.
4. Выставить ширину frame'а 393, высоту 852.
5. Lottie → экспортнуть через LottieFiles плагин отдельно, вставить на место `lottie-player` placeholders.
6. Пробежаться по переменным: скопировать CSS `:root { --bg, --primary, ... }` в Figma Variables (можно через figma-mcp-go `create_variable_collection` + `create_variable` — тут он сильно экономит время).
7. Если осталось >20 % грязи → повторить через Builder.io, сравнить, взять лучший.

## Disputed / Unverified Claims

- html.to.design реально парсит localStorage/private-pages — 🟡 заявлено вендором, но 10-импортного free-тира хватит проверить самому перед покупкой.
- figma-mcp-go обходит rate-limit REST API через plugin-bridge — 🟡 подтверждено README автора, независимых бенчмарков не нашёл.
- Builder.io 80–90 % fidelity — 🔴 self-reported; Reddit-отзывы 2025 скорее 60–75 % на вложенном flex.

## Research Metadata
- **Sources**: 22
- **Parallel agents**: 3 (плагины / figma-mcp-go / Claude Code skills)
- **Passes**: 5
- **Depth**: Deep

## Sources

1. [html.to.design Figma plugin](https://www.figma.com/community/plugin/1159123024924461424/html-to-design-by-divriots-import-websites-to-figma-designs-web-html-css) — основной кандидат
2. [html.to.design — Pro plan & limits](https://html.to.design/docs/pro-plan/)
3. [html.to.design — mobile import](https://html.to.design/docs/capturing-mobile-version)
4. [html.to.design — Import code blog](https://html.to.design/blog/new-feature-import-code-in-figma/)
5. [html.to.design — From Claude Code to Figma](https://html.to.design/blog/from-claude-code-to-figma/)
6. [html.to.design MCP docs](https://html.to.design/docs/mcp-tab/)
7. [html2design.com native plugin](https://html2design.com/)
8. [Builder.io HTML-to-Figma blog](https://www.builder.io/blog/html-to-design)
9. [Builder.io Figma plugin docs](https://www.builder.io/c/docs/builder-figma-plugin)
10. [Builder.io Visual Copilot CLI](https://www.builder.io/blog/visual-copilot-cli)
11. [Figma Dev Mode MCP Server announcement](https://www.figma.com/blog/introducing-figma-mcp-server/)
12. [Figma MCP developer docs](https://developers.figma.com/docs/figma-mcp-server/)
13. [Figma: Introducing Claude Code to Figma](https://www.figma.com/blog/introducing-claude-code-to-figma/)
14. [Figma plugin on claude.com](https://claude.com/plugins/figma)
15. [vkhanhqui/figma-mcp-go (GitHub)](https://github.com/vkhanhqui/figma-mcp-go) — уже установлен
16. [GLips/Figma-Context-MCP (Framelink, read-only)](https://github.com/GLips/Figma-Context-MCP)
17. [markacianfrani/claude-code-figma](https://github.com/markacianfrani/claude-code-figma)
18. [Figma Plugin API — createFrame](https://www.figma.com/plugin-docs/api/properties/figma-createframe/)
19. [r/FigmaDesign — Figma-to-Code experiences 2025](https://www.reddit.com/r/FigmaDesign/comments/1k04729/figmatocode_experiences/)
20. [r/FigmaDesign — html.to.design Show Reddit](https://www.reddit.com/r/FigmaDesign/comments/1k0qykn/show_reddit_htmltofigdesign_convert_any/)
21. [r/Frontend — Figma-to-code skepticism](https://www.reddit.com/r/Frontend/comments/1nickcg/have_you_tried_any_figmatocode_tools_and_got/)
22. [LottieFiles for Figma plugin](https://www.figma.com/community/plugin/809860933081065308/lottiefiles-discover-create-export-lottie-animations)
