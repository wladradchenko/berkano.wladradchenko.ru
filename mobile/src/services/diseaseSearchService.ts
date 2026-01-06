import * as ort from 'onnxruntime-react-native';
import { getModel } from './modelManager';
import { findTopKSimilar } from './embeddingsService';

export interface DiseaseSearchResult {
  index: number;
  distance: number;
  caption: string;
}

/**
 * Выполняет поиск похожих изображений болезней
 * @param imageTensor Тензор изображения формата [1, 3, 224, 224]
 * @param topK Количество результатов
 * @returns Массив результатов поиска
 */
export async function searchDisease(
  imageTensor: Float32Array,
  topK: number = 10,
): Promise<DiseaseSearchResult[]> {
  try {
    // Загружаем модель (автоматически через modelManager)
    const modelInfo = await getModel('disease');
    
    if (!modelInfo.session) {
      throw new Error('Disease model session is null');
    }

    // Создаем тензор для входных данных
    const inputTensor = new ort.Tensor('float32', imageTensor, [1, 3, 224, 224]);

    // Выполняем инференс
    const feeds = { [modelInfo.inputName]: inputTensor };
    const results = await modelInfo.session.run(feeds);

    // Получаем выходной тензор
    const outputTensor = results[modelInfo.outputName];
    const embedding = outputTensor.data as Float32Array;

    // Нормализуем эмбеддинг
    const norm = Math.sqrt(
      Array.from(embedding).reduce((sum, val) => sum + val * val, 0),
    );
    const normalizedEmbedding = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      normalizedEmbedding[i] = embedding[i] / norm;
    }

    // Ищем похожие изображения
    const searchResults = await findTopKSimilar(normalizedEmbedding, topK);

    return searchResults;
  } catch (error) {
    console.error('Error searching disease:', error);
    throw error;
  }
}

