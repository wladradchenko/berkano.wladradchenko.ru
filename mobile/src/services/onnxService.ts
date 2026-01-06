import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';

async function copyAssetIfNeeded(assetName: string, folder: string): Promise<string> {
  const localPath = `${RNFS.DocumentDirectoryPath}/${assetName}`;

  const exists = await RNFS.exists(localPath);
  if (!exists) {
    try {
      if (Platform.OS === 'android') {
        // Android: читаем из assets
        // Путь должен быть относительно папки assets: folder/assetName
        const assetPath = `${folder}/${assetName}`;
        console.log(`Attempting to copy from assets: ${assetPath} to ${localPath}`);
        await RNFS.copyFileAssets(assetPath, localPath);
        console.log(`Successfully copied ${assetName} to local FS: ${localPath}`);
      } else {
        // iOS: читаем из main bundle
        const source = `${RNFS.MainBundlePath}/${folder}/${assetName}`;
        console.log(`Attempting to copy from bundle: ${source} to ${localPath}`);
        await RNFS.copyFile(source, localPath);
        console.log(`Successfully copied ${assetName} to local FS: ${localPath}`);
      }
    } catch (err: any) {
      console.error(`Failed to copy ${assetName} from ${folder}/${assetName}:`, err);
      console.error(`Error details:`, {
        message: err?.message,
        code: err?.code,
        platform: Platform.OS,
        localPath,
        assetPath: `${folder}/${assetName}`,
      });
      // Пробуем альтернативный путь без папки (на случай если файл в корне assets)
      if (Platform.OS === 'android') {
        try {
          console.log(`Trying alternative path: ${assetName}`);
          await RNFS.copyFileAssets(assetName, localPath);
          console.log(`Successfully copied using alternative path`);
        } catch (altErr) {
          console.error(`Alternative path also failed:`, altErr);
          throw new Error(`Не удалось скопировать ${assetName}. Проверьте, что файл находится в assets/${folder}/`);
        }
      } else {
        throw err;
      }
    }
  } else {
    console.log(`${assetName} already exists at ${localPath}`);
  }

  return localPath;
}

export async function prepareModelAssets() {
  const modelPath = await copyAssetIfNeeded('scold_image_only_fp16.onnx', 'models');
  const embeddingsPath = await copyAssetIfNeeded('embeddings.bin', 'files');
  const captionsPath = await copyAssetIfNeeded('captions.json', 'files');
  const classMappingPath = await copyAssetIfNeeded('class_mapping.txt', 'files');
  const speciesMappingPath = await copyAssetIfNeeded('species_id_to_name.txt', 'files');

  return { 
    modelPath, 
    embeddingsPath, 
    captionsPath,
    classMappingPath,
    speciesMappingPath,
  };
}

export interface ModelSession {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
}

let modelSession: ModelSession | null = null;

/**
 * Инициализирует ONNX модель с приоритетом GPU/NPU через NNAPI
 */
export async function initializeModel(): Promise<ModelSession> {
  if (modelSession) {
    return modelSession;
  }

  try {
    // Используем путь из prepareModelAssets
    const { modelPath } = await prepareModelAssets();
    
    // Используем локальный путь к модели
    const modelUri = modelPath;
    
    // Настройки сессии с приоритетом NNAPI (GPU/NPU)
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['nnapi', 'cpu'], // NNAPI для GPU/NPU, CPU как fallback
      graphOptimizationLevel: 'all',
    };

    // Создаем сессию
    // onnxruntime-react-native может работать с путями к файлам напрямую
    const session = await ort.InferenceSession.create(modelUri, sessionOptions);

    // Получаем имена входов и выходов
    const inputNames = session.inputNames;
    const outputNames = session.outputNames;

    if (inputNames.length === 0 || outputNames.length === 0) {
      throw new Error('Модель не имеет входов или выходов');
    }

    modelSession = {
      session,
      inputName: inputNames[0], // Предполагаем первый вход
      outputName: outputNames[0], // Предполагаем первый выход
    };

    console.log('Модель успешно загружена');
    console.log('Input name:', modelSession.inputName);
    console.log('Output name:', modelSession.outputName);

    return modelSession;
  } catch (error) {
    console.error('Ошибка при загрузке модели:', error);
    throw error;
  }
}

/**
 * Выполняет инференс модели на изображении
 * @param imageTensor Тензор изображения формата [1, 3, 224, 224]
 * @returns Эмбеддинг изображения
 */
export async function runInference(
  imageTensor: Float32Array,
): Promise<Float32Array> {
  if (!modelSession) {
    throw new Error('Модель не инициализирована. Вызовите initializeModel() сначала.');
  }

  try {
    // Создаем тензор для входных данных
    // Формат: [batch, channels, height, width] = [1, 3, 224, 224]
    const inputTensor = new ort.Tensor('float32', imageTensor, [1, 3, 224, 224]);

    // Выполняем инференс
    const feeds = { [modelSession.inputName]: inputTensor };
    const results = await modelSession.session.run(feeds);

    // Получаем выходной тензор
    const outputTensor = results[modelSession.outputName];
    const embedding = outputTensor.data as Float32Array;

    // Нормализуем эмбеддинг
    const norm = Math.sqrt(
      Array.from(embedding).reduce((sum, val) => sum + val * val, 0),
    );
    const normalizedEmbedding = new Float32Array(embedding.length);
    for (let i = 0; i < embedding.length; i++) {
      normalizedEmbedding[i] = embedding[i] / norm;
    }

    return normalizedEmbedding;
  } catch (error) {
    console.error('Ошибка при выполнении инференса:', error);
    throw error;
  }
}

/**
 * Освобождает ресурсы модели
 */
export function disposeModel(): void {
  if (modelSession) {
    modelSession.session.release();
    modelSession = null;
  }
}

