import { Image } from 'react-native';
import ImageResizer from 'react-native-image-resizer';
import { NativeModules } from 'react-native';


export interface PreprocessedImage {
  data: Float32Array;
  width: number;
  height: number;
}

/**
 * Предобработка изображения для модели ONNX
 * Изменяет размер до 224x224 и конвертирует в тензор
 */
export async function preprocessImage(
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
    const imageData = await loadImageAsTensor(resizedImage.uri);
    
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
 * Загружает изображение и конвертирует в тензор формата [1, 3, 224, 224]
 * Значения нормализованы в диапазоне [0, 1] (как в torchvision.transforms.ToTensor)
 * 
 * Нативный модуль возвращает данные в формате CHW [3, H, W]
 * Нужно преобразовать в NCHW [1, 3, 224, 224]
 */
async function loadImageAsTensor(imageUri: string): Promise<Float32Array> {
  try {
    // Проверяем, есть ли нативный модуль для декодирования изображений
    const ImageDecoder = NativeModules.ImageDecoder;
    if (!ImageDecoder) {
      throw new Error('ImageDecoder native module not available');
    }

    console.log('Calling ImageDecoder.decodeToTensor with URI:', imageUri);
    
    // Вызываем нативный метод (он возвращает Promise)
    // Нативный модуль возвращает массив в формате CHW: [R, R, ..., G, G, ..., B, B, ...]
    const chwTensor = await ImageDecoder.decodeToTensor(
      imageUri, 
      224, // targetWidth
      224  // targetHeight
    );
    
    console.log('Received tensor from native module, type:', typeof chwTensor, 'length:', chwTensor?.length);
    
    // Конвертируем в обычный массив, если это еще не массив
    const tensorArray = Array.isArray(chwTensor) ? chwTensor : Array.from(chwTensor);
    
    // Размеры
    const height = 224;
    const width = 224;
    const channels = 3;
    const batchSize = 1;
    
    // Создаем тензор в формате NCHW [1, 3, 224, 224]
    const nchwTensor = new Float32Array(batchSize * channels * height * width);
    
    // Преобразуем из CHW в NCHW
    // CHW формат: [R...R, G...G, B...B] где каждый канал имеет H*W элементов
    // NCHW формат: [N=0, C=0, H, W], [N=0, C=1, H, W], [N=0, C=2, H, W]
    const channelSize = height * width;
    
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
          // Индекс в CHW формате
          const chwIndex = c * channelSize + h * width + w;
          // Индекс в NCHW формате [1, 3, 224, 224] (для batch=0)
          const nchwIndex = c * channelSize + h * width + w;
          nchwTensor[nchwIndex] = Number(tensorArray[chwIndex]);
        }
      }
    }
    
    console.log('Converted to NCHW tensor, shape: [1, 3, 224, 224], length:', nchwTensor.length);
    
    return nchwTensor;
  } catch (error) {
    console.error('Ошибка при загрузке изображения как тензора:', error);
    throw error;
  }
}

