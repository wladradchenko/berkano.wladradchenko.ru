import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Platform } from 'react-native';
import { getModel } from './modelManager';

export interface PlantClassificationResult {
  speciesName: string;
  probability: number;
}

export interface AgeClassificationResult {
  age: number;
  leafCount: number;
}

let classMapping: Map<number, string> | null = null;
let speciesMapping: Map<string, string> | null = null;

/**
 * Копирует файл из assets
 */
async function copyAssetIfNeeded(assetName: string, folder: string): Promise<string> {
  const localPath = `${RNFS.DocumentDirectoryPath}/${assetName}`;
  const exists = await RNFS.exists(localPath);
  
  if (!exists) {
    try {
      if (Platform.OS === 'android') {
        await RNFS.copyFileAssets(`${folder}/${assetName}`, localPath);
      } else {
        const source = `${RNFS.MainBundlePath}/${folder}/${assetName}`;
        await RNFS.copyFile(source, localPath);
      }
    } catch (err) {
      console.error(`Failed to copy ${assetName}:`, err);
      throw err;
    }
  }
  
  return localPath;
}

/**
 * Загружает class_mapping.txt
 */
async function loadClassMapping(): Promise<Map<number, string>> {
  if (classMapping) {
    return classMapping;
  }

  try {
    const classMappingPath = await copyAssetIfNeeded('class_mapping.txt', 'files');
    const content = await RNFS.readFile(classMappingPath, 'utf8');
    const lines = content.split('\n').filter(line => line.trim());
    
    const mapping = new Map<number, string>();
    lines.forEach((line, index) => {
      mapping.set(index, line.trim());
    });

    classMapping = mapping;
    console.log(`Loaded ${mapping.size} classes`);
    return mapping;
  } catch (error) {
    console.error('Error loading class mapping:', error);
    throw error;
  }
}

/**
 * Загружает species_id_to_name.txt
 */
async function loadSpeciesMapping(): Promise<Map<string, string>> {
  if (speciesMapping) {
    return speciesMapping;
  }

  try {
    const speciesMappingPath = await copyAssetIfNeeded('species_id_to_name.txt', 'files');
    const content = await RNFS.readFile(speciesMappingPath, 'utf8');
    const lines = content.split('\n').filter(line => line.trim());
    
    const mapping = new Map<string, string>();
    // Пропускаем первую строку (заголовок)
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];
      // Парсим CSV: "species_id";"species"
      // Используем регулярное выражение для правильного парсинга с учетом кавычек
      const match = line.match(/"([^"]+)";"([^"]+)"/);
      if (match) {
        const speciesId = match[1];
        const speciesName = match[2];
        mapping.set(speciesId, speciesName);
      }
    }

    speciesMapping = mapping;
    console.log(`Loaded ${mapping.size} species`);
    return mapping;
  } catch (error) {
    console.error('Error loading species mapping:', error);
    throw error;
  }
}


/**
 * Классифицирует возвраста по изображению
 * @param imageTensor Тензор изображения формата [1, 3, 224, 224]
 * @param topK Количество топ результатов
 * @returns Массив результатов классификации
 */
export async function classifyAge(
  imageTensor: Float32Array,
): Promise<AgeClassificationResult[]> {
  try {
    // Загружаем модель
    const modelInfo = await getModel('age');

    if (!modelInfo.session) {
      throw new Error('Age model session is null');
    }

    // Создаем тензор для входных данных
    const inputTensor = new ort.Tensor('float32', imageTensor, [1, 3, 224, 224]);

    // Выполняем инференс
    const feeds = { [modelInfo.inputName]: inputTensor };
    const results = await modelInfo.session.run(feeds);

    // Получаем выходной тензор
    const age = Math.floor((results.age.data as Float32Array)[0]);
    const leafCount = Math.floor((results.count.data as Float32Array)[0]);

    return [{ age, leafCount }];
  } catch (error) {
    console.error('Error classifying plant:', error);
    throw error;
  }
}

/**
 * Классифицирует растение по изображению
 * @param imageTensor Тензор изображения формата [1, 3, 518, 518]
 * @param topK Количество топ результатов
 * @returns Массив результатов классификации
 */
export async function classifyPlant(
  imageTensor: Float32Array,
  topK: number = 5,
): Promise<PlantClassificationResult[]> {
  try {
    // Загружаем модель
    const modelInfo = await getModel('plant');
    
    if (!modelInfo.session) {
      throw new Error('Plant model session is null');
    }

    // Загружаем маппинги
    const classMap = await loadClassMapping();
    const speciesMap = await loadSpeciesMapping();

    // Создаем тензор для входных данных
    // Формат: [batch, channels, height, width] = [1, 3, 518, 518]
    const inputTensor = new ort.Tensor('float32', imageTensor, [1, 3, 518, 518]);

    // Выполняем инференс
    const feeds = { [modelInfo.inputName]: inputTensor };
    const results = await modelInfo.session.run(feeds);

    // Получаем выходной тензор
    const outputTensor = results[modelInfo.outputName];
    const probabilities = outputTensor.data as Float32Array;

    // Находим топ-K индексов
    const indexed = Array.from(probabilities)
      .map((prob, index) => ({ index, prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, topK);

    // Формируем результаты
    const classificationResults: PlantClassificationResult[] = [];

    for (const { index, prob } of indexed) {
      const classId = classMap.get(index);
      if (classId) {
        const speciesName = speciesMap.get(classId) || classId;
        classificationResults.push({
          speciesName,
          probability: prob,
        });
      }
    }

    return classificationResults;
  } catch (error) {
    console.error('Error classifying plant:', error);
    throw error;
  }
}

