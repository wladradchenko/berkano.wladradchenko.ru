# Мультиязычность (i18n)

Приложение поддерживает английский и русский языки.

## Структура

```
src/i18n/
├── config.ts          # Конфигурация i18n
└── locales/
    ├── en.json        # Английские переводы
    └── ru.json        # Русские переводы
```

## Использование в компонентах

```typescript
import { useTranslation } from 'react-i18next';

function MyComponent() {
  const { t, i18n } = useTranslation();
  
  // Использование перевода
  return <Text>{t('app.title')}</Text>;
  
  // Переключение языка
  const toggleLanguage = () => {
    i18n.changeLanguage(i18n.language === 'ru' ? 'en' : 'ru');
  };
}
```

## Добавление новых переводов

1. Добавьте ключ в `src/i18n/locales/en.json`
2. Добавьте тот же ключ в `src/i18n/locales/ru.json`
3. Используйте `t('your.key')` в компонентах

## Сохранение языка

Выбранный язык автоматически сохраняется в AsyncStorage и восстанавливается при следующем запуске приложения.

