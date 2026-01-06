import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';
import en from './locales/en.json';
import ru from './locales/ru.json';

const LANGUAGE_KEY = '@app_language';

// Инициализируем i18n
i18n
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        translation: en,
      },
      ru: {
        translation: ru,
      },
    },
    lng: 'ru', // Язык по умолчанию
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false, // React уже экранирует значения
    },
  });

// Загружаем сохраненный язык и применяем его
AsyncStorage.getItem(LANGUAGE_KEY).then((savedLanguage) => {
  if (savedLanguage && (savedLanguage === 'en' || savedLanguage === 'ru')) {
    i18n.changeLanguage(savedLanguage);
  }
});

// Сохраняем язык при изменении
i18n.on('languageChanged', (lng) => {
  AsyncStorage.setItem(LANGUAGE_KEY, lng);
});

export default i18n;

