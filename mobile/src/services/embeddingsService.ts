import { NativeModules } from 'react-native';
import { prepareModelAssets } from './onnxService';

export interface SearchResult {
  index: number;
  distance: number;
  caption: string;
}

const FaissSearch = NativeModules.FaissSearch;

let isInitialized = false;

/**
 * Инициализирует нативный модуль поиска
 * Загружает embeddings и captions в нативную память
 */
export async function initializeSearch(): Promise<void> {
  if (isInitialized) {
    return;
  }

  try {
    if (!FaissSearch) {
      throw new Error('FaissSearch native module not available');
    }

    // Получаем пути к файлам
    const { embeddingsPath, captionsPath } = await prepareModelAssets();

    // Загружаем данные последовательно с небольшими задержками для освобождения памяти
    console.log('Loading embeddings into native memory...');
    try {
      await FaissSearch.loadEmbeddings(embeddingsPath);
      console.log('Embeddings loaded successfully');
    } catch (error: any) {
      if (error?.code === 'OUT_OF_MEMORY') {
        throw new Error('Недостаточно памяти для загрузки embeddings. Файл слишком большой.');
      }
      throw error;
    }

    // Небольшая задержка для освобождения памяти после загрузки embeddings
    await new Promise(resolve => setTimeout(resolve, 100));

    console.log('Loading captions...');
    try {
      await FaissSearch.loadCaptions(captionsPath);
      console.log('Captions loaded successfully');
    } catch (error: any) {
      if (error?.code === 'OUT_OF_MEMORY') {
        throw new Error('Недостаточно памяти для загрузки captions. Файл слишком большой.');
      }
      throw error;
    }

    isInitialized = true;
  } catch (error) {
    console.error('Ошибка при инициализации поиска:', error);
    throw error;
  }
}

/**
 * Находит топ-K ближайших векторов к запросу
 * Поиск выполняется в нативной памяти, в JS передаются только результаты
 * @param queryEmbedding Эмбеддинг запроса
 * @param topK Количество результатов (по умолчанию 10)
 * @returns Массив результатов поиска
 */
export async function findTopKSimilar(
  queryEmbedding: Float32Array,
  topK: number = 10,
): Promise<SearchResult[]> {
  try {
    if (!isInitialized) {
      await initializeSearch();
    }

    if (!FaissSearch) {
      throw new Error('FaissSearch native module not available');
    }

    // Конвертируем Float32Array в обычный массив для передачи в нативный модуль
    const queryArray = Array.from(queryEmbedding);

    // Выполняем поиск в нативной памяти
    const results = await FaissSearch.search(queryArray, topK);

    // Конвертируем результаты в нужный формат
    const searchResults: SearchResult[] = results.map((result: any) => ({
      index: result.index,
      distance: result.distance,
      caption: result.caption,
    }));

    return searchResults;
  } catch (error) {
    console.error('Ошибка при поиске:', error);
    throw error;
  }
}

/**
 * Проверяет, загружены ли данные
 */
export async function isSearchInitialized(): Promise<boolean> {
  if (!FaissSearch) {
    return false;
  }

  try {
    const status = await FaissSearch.isLoaded();
    return status.embeddingsLoaded && status.captionsLoaded;
  } catch (error) {
    return false;
  }
}

/**
 * Очищает кэш из нативной памяти
 */
export async function clearCache(): Promise<void> {
  if (FaissSearch) {
    await FaissSearch.clearCache();
    isInitialized = false;
  }
}
