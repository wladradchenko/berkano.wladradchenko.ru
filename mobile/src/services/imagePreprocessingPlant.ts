import { Image } from 'react-native';
import ImageResizer from 'react-native-image-resizer';
import { NativeModules } from 'react-native';

export interface PreprocessedImage {
  data: Float32Array;
  width: number;
  height: number;
}

/**
 * Предобработка изображения для модели классификации растений
 * Изменяет размер до 224x224 и применяет нормализацию ImageNet
 */
export async function preprocessImageForAge(
  imageUri: string,
): Promise<PreprocessedImage> {
  try {
    // Изменяем размер изображения до 224x224
    const resizedImage = await ImageResizer.createResizedImage(
      imageUri,
      224,
      224,
      'JPEG',
      100,
      0,
      undefined,
      false,
      {
        mode: 'contain',
        onlyScaleDown: false,
      },
    );

    // Загружаем изображение и конвертируем в тензор
    const imageData = await loadImageAsTensorForAge(resizedImage.uri);
    
    return {
      data: imageData,
      width: 224,
      height: 224,
    };
  } catch (error) {
    console.error('Ошибка при предобработке изображения:', error);
    throw error;
  }
}


/**
 * Предобработка изображения для модели классификации растений
 * Изменяет размер до 518x518 и применяет нормализацию ImageNet
 */
export async function preprocessImageForPlant(
  imageUri: string,
): Promise<PreprocessedImage> {
  try {
    // Изменяем размер изображения до 518x518
    const resizedImage = await ImageResizer.createResizedImage(
      imageUri,
      518,
      518,
      'JPEG',
      100,
      0,
      undefined,
      false,
      {
        mode: 'contain',
        onlyScaleDown: false,
      },
    );

    // Загружаем изображение и конвертируем в тензор
    const imageData = await loadImageAsTensorForPlant(resizedImage.uri);
    
    return {
      data: imageData,
      width: 518,
      height: 518,
    };
  } catch (error) {
    console.error('Ошибка при предобработке изображения:', error);
    throw error;
  }
}


/**
 * Загружает изображение и конвертирует в тензор формата [1, 3, 224, 224]
 * Применяет нормализацию ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
 */
async function loadImageAsTensorForAge(imageUri: string): Promise<Float32Array> {
  try {
    const ImageDecoder = NativeModules.ImageDecoder;
    if (!ImageDecoder) {
      throw new Error('ImageDecoder native module not available');
    }

    console.log('Calling ImageDecoder.decodeToTensor for plant classification...');
    
    // Вызываем нативный метод (он возвращает массив в формате CHW)
    const chwTensor = await ImageDecoder.decodeToTensor(
      imageUri, 
      224, // targetWidth
      224  // targetHeight
    );
    
    console.log('Received tensor from native module, length:', chwTensor?.length);
    
    // Конвертируем в обычный массив
    const tensorArray = Array.isArray(chwTensor) ? chwTensor : Array.from(chwTensor);
    
    // Размеры
    const height = 224;
    const width = 224;
    const channels = 3;
    const batchSize = 1;
    
    // Создаем тензор в формате NCHW [1, 3, 224, 224]
    const nchwTensor = new Float32Array(batchSize * channels * height * width);
    
    // ImageNet нормализация: mean и std
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    // Преобразуем из CHW в NCHW и применяем нормализацию
    const channelSize = height * width;
    
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          // Индекс в CHW формате
          const chwIndex = c * channelSize + h * width + w;
          // Индекс в NCHW формате [1, 3, 518, 518]
          const nchwIndex = c * channelSize + h * width + w;
          
          // Значение уже нормализовано в [0, 1] из нативного модуля
          // Применяем ImageNet нормализацию: (x - mean) / std
          const normalizedValue = (Number(tensorArray[chwIndex]) - mean[c]) / std[c];
          nchwTensor[nchwIndex] = normalizedValue;
        }
      }
    }
    
    console.log('Converted to NCHW tensor with ImageNet normalization, shape: [1, 3, 224, 224]');
    
    return nchwTensor;
  } catch (error) {
    console.error('Ошибка при загрузке изображения как тензора:', error);
    throw error;
  }
}



/**
 * Загружает изображение и конвертирует в тензор формата [1, 3, 518, 518]
 * Применяет нормализацию ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
 */
async function loadImageAsTensorForPlant(imageUri: string): Promise<Float32Array> {
  try {
    const ImageDecoder = NativeModules.ImageDecoder;
    if (!ImageDecoder) {
      throw new Error('ImageDecoder native module not available');
    }

    console.log('Calling ImageDecoder.decodeToTensor for plant classification...');
    
    // Вызываем нативный метод (он возвращает массив в формате CHW)
    const chwTensor = await ImageDecoder.decodeToTensor(
      imageUri, 
      518, // targetWidth
      518  // targetHeight
    );
    
    console.log('Received tensor from native module, length:', chwTensor?.length);
    
    // Конвертируем в обычный массив
    const tensorArray = Array.isArray(chwTensor) ? chwTensor : Array.from(chwTensor);
    
    // Размеры
    const height = 518;
    const width = 518;
    const channels = 3;
    const batchSize = 1;
    
    // Создаем тензор в формате NCHW [1, 3, 518, 518]
    const nchwTensor = new Float32Array(batchSize * channels * height * width);
    
    // ImageNet нормализация: mean и std
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    // Преобразуем из CHW в NCHW и применяем нормализацию
    const channelSize = height * width;
    
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          // Индекс в CHW формате
          const chwIndex = c * channelSize + h * width + w;
          // Индекс в NCHW формате [1, 3, 518, 518]
          const nchwIndex = c * channelSize + h * width + w;
          
          // Значение уже нормализовано в [0, 1] из нативного модуля
          // Применяем ImageNet нормализацию: (x - mean) / std
          const normalizedValue = (Number(tensorArray[chwIndex]) - mean[c]) / std[c];
          nchwTensor[nchwIndex] = normalizedValue;
        }
      }
    }
    
    console.log('Converted to NCHW tensor with ImageNet normalization, shape: [1, 3, 518, 518]');
    
    return nchwTensor;
  } catch (error) {
    console.error('Ошибка при загрузке изображения как тензора:', error);
    throw error;
  }
}

