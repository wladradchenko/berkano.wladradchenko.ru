import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';

export type ModelType = 'disease' | 'plant' | 'age';

export interface ModelInfo {
  type: ModelType;
  path: string;
  session: ort.InferenceSession | null;
  inputName: string;
  outputName: string;
  isLoaded: boolean;
  lastUsed: number;
}

// Максимальное количество моделей в памяти одновременно
const MAX_MODELS_IN_MEMORY = 2;

// Кэш моделей
const modelCache: Map<ModelType, ModelInfo> = new Map();

/**
 * Копирует файл из assets в локальную файловую систему
 */
async function copyAssetIfNeeded(assetName: string, folder: string): Promise<string> {
  const localPath = `${RNFS.DocumentDirectoryPath}/${assetName}`;

  const exists = await RNFS.exists(localPath);
  if (!exists) {
    try {
      if (Platform.OS === 'android') {
        const assetPath = `${folder}/${assetName}`;
        console.log(`Copying ${assetName} from assets...`);
        await RNFS.copyFileAssets(assetPath, localPath);
        console.log(`Successfully copied ${assetName}`);
      } else {
        const source = `${RNFS.MainBundlePath}/${folder}/${assetName}`;
        await RNFS.copyFile(source, localPath);
      }
    } catch (err: any) {
      console.error(`Failed to copy ${assetName}:`, err);
      if (Platform.OS === 'android') {
        try {
          await RNFS.copyFileAssets(assetName, localPath);
        } catch (altErr) {
          throw new Error(`Не удалось скопировать ${assetName}`);
        }
      } else {
        throw err;
      }
    }
  }

  return localPath;
}

/**
 * Выгружает наименее используемую модель из памяти
 */
function unloadLeastUsedModel(): void {
  let leastUsed: ModelInfo | null = null;
  let leastUsedTime = Date.now();

  for (const model of modelCache.values()) {
    if (model.isLoaded && model.lastUsed < leastUsedTime) {
      leastUsed = model;
      leastUsedTime = model.lastUsed;
    }
  }

  if (leastUsed) {
    console.log(`Unloading model: ${leastUsed.type}`);
    if (leastUsed.session) {
      leastUsed.session.release();
      leastUsed.session = null;
      leastUsed.isLoaded = false;
    }
  }
}

/**
 * Загружает модель в память
 */
async function loadModel(modelInfo: ModelInfo): Promise<void> {
  if (modelInfo.isLoaded && modelInfo.session) {
    modelInfo.lastUsed = Date.now();
    return;
  }

  // Проверяем, не превышен ли лимит моделей в памяти
  const loadedModels = Array.from(modelCache.values()).filter(m => m.isLoaded);
  if (loadedModels.length >= MAX_MODELS_IN_MEMORY) {
    console.log('Memory limit reached, unloading least used model...');
    unloadLeastUsedModel();
  }

  try {
    console.log(`Loading model: ${modelInfo.type}`);
    
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['nnapi', 'cpu'],
      graphOptimizationLevel: 'all',
    };

    const session = await ort.InferenceSession.create(modelInfo.path, sessionOptions);

    const inputNames = session.inputNames;
    const outputNames = session.outputNames;

    if (inputNames.length === 0 || outputNames.length === 0) {
      throw new Error('Модель не имеет входов или выходов');
    }

    modelInfo.session = session;
    modelInfo.inputName = inputNames[0];
    modelInfo.outputName = outputNames[0];
    modelInfo.isLoaded = true;
    modelInfo.lastUsed = Date.now();

    console.log(`Model ${modelInfo.type} loaded successfully`);
  } catch (error) {
    console.error(`Error loading model ${modelInfo.type}:`, error);
    throw error;
  }
}

/**
 * Инициализирует систему управления моделями
 */
export async function initializeModelManager(): Promise<void> {
  // Подготавливаем пути к моделям
  const diseaseModelPath = await copyAssetIfNeeded('disease_detection.onnx', 'models');
  const plantModelPath = await copyAssetIfNeeded('plant_classification.onnx', 'models');
  const ageModelPath = await copyAssetIfNeeded('plant_analysis.onnx', 'models');

  // Регистрируем модели
  modelCache.set('disease', {
    type: 'disease',
    path: diseaseModelPath,
    session: null,
    inputName: '',
    outputName: '',
    isLoaded: false,
    lastUsed: 0,
  });

  modelCache.set('plant', {
    type: 'plant',
    path: plantModelPath,
    session: null,
    inputName: '',
    outputName: '',
    isLoaded: false,
    lastUsed: 0,
  });

  modelCache.set('age', {
    type: 'age',
    path: ageModelPath,
    session: null,
    inputName: '',
    outputName: '',
    isLoaded: false,
    lastUsed: 0,
  });

  console.log('Model manager initialized');
}

/**
 * Получает модель (загружает если нужно)
 */
export async function getModel(modelType: ModelType): Promise<ModelInfo> {
  const modelInfo = modelCache.get(modelType);
  if (!modelInfo) {
    throw new Error(`Model ${modelType} not found`);
  }

  await loadModel(modelInfo);
  return modelInfo;
}

/**
 * Выгружает модель из памяти
 */
export function unloadModel(modelType: ModelType): void {
  const modelInfo = modelCache.get(modelType);
  if (modelInfo && modelInfo.session) {
    console.log(`Unloading model: ${modelType}`);
    modelInfo.session.release();
    modelInfo.session = null;
    modelInfo.isLoaded = false;
  }
}

/**
 * Выгружает все модели
 */
export function unloadAllModels(): void {
  for (const modelType of modelCache.keys()) {
    unloadModel(modelType);
  }
}

/**
 * Проверяет, загружена ли модель
 */
export function isModelLoaded(modelType: ModelType): boolean {
  const modelInfo = modelCache.get(modelType);
  return modelInfo?.isLoaded ?? false;
}

