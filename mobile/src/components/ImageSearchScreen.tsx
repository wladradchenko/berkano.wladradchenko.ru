import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  TextInput,
  Alert,
} from 'react-native';
import { launchImageLibrary, ImagePickerResponse } from 'react-native-image-picker';
import { initializeModel, runInference } from '../services/onnxService';
import { findTopKSimilar, SearchResult, initializeSearch } from '../services/embeddingsService';
import { preprocessImage } from '../services/imagePreprocessing';

const DEFAULT_TOP_K = 10;

export default function ImageSearchScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [searchReady, setSearchReady] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [initProgress, setInitProgress] = useState('');
  const [topK, setTopK] = useState<string>(DEFAULT_TOP_K.toString());

  useEffect(() => {
    // Инициализируем модель и поиск при загрузке компонента
    // Загружаем последовательно с задержками для освобождения памяти
    const init = async () => {
      try {
        setInitProgress('Загрузка модели...');
        console.log('Initializing model...');
        await initializeModel();
        setModelReady(true);
        setInitProgress('Модель загружена. Загрузка данных для поиска...');
        console.log('Модель готова к использованию');

        // Небольшая задержка перед загрузкой поиска
        await new Promise<void>(resolve => setTimeout(() => resolve(), 500));

        setInitProgress('Загрузка embeddings в нативную память...');
        console.log('Initializing search...');
        await initializeSearch();
        setSearchReady(true);
        setInitProgress('Готово!');
        console.log('Поиск готов к использованию');
        
        // Скрываем прогресс через секунду
        setTimeout(() => {
          setInitializing(false);
          setInitProgress('');
        }, 1000);
      } catch (error: any) {
        console.error('Ошибка инициализации:', error);
        setInitializing(false);
        const errorMessage = error?.message || String(error);
        Alert.alert(
          'Ошибка инициализации',
          `Не удалось инициализировать: ${errorMessage}\n\nВозможные причины:\n- Недостаточно памяти устройства\n- Файлы слишком большие (>200MB)\n- Проблема с загрузкой файлов\n\nПопробуйте:\n- Закрыть другие приложения\n- Перезапустить приложение`,
        );
      }
    };
    init();
  }, []);

  const selectImage = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 1,
      },
      (response: ImagePickerResponse) => {
        if (response.assets && response.assets[0]) {
          setSelectedImage(response.assets[0].uri || null);
          setResults([]);
        }
      },
    );
  };

  const searchSimilar = async () => {
    if (!selectedImage) {
      Alert.alert('Ошибка', 'Выберите изображение');
      return;
    }

    if (!modelReady) {
      Alert.alert('Ошибка', 'Модель еще не загружена');
      return;
    }

    setLoading(true);
    setResults([]);

    try {
      // Предобработка изображения
      console.log('Предобработка изображения...');
      const preprocessed = await preprocessImage(selectedImage);

      // Выполнение инференса
      console.log('Выполнение инференса...');
      const embedding = await runInference(preprocessed.data);

      // Поиск похожих векторов
      console.log('Поиск похожих векторов...');
      const topKValue = parseInt(topK, 10) || DEFAULT_TOP_K;
      const searchResults = await findTopKSimilar(embedding, topKValue);

      setResults(searchResults);
      console.log(`Найдено ${searchResults.length} результатов`);
    } catch (error) {
      console.error('Ошибка при поиске:', error);
      Alert.alert('Ошибка', `Не удалось выполнить поиск: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Поиск похожих изображений</Text>

      {/* Выбор количества результатов */}
      <View style={styles.topKContainer}>
        <Text style={styles.label}>Количество результатов:</Text>
        <TextInput
          style={styles.input}
          value={topK}
          onChangeText={setTopK}
          keyboardType="numeric"
          placeholder={DEFAULT_TOP_K.toString()}
        />
      </View>

      {/* Выбор изображения */}
      <TouchableOpacity style={styles.button} onPress={selectImage}>
        <Text style={styles.buttonText}>Выбрать изображение</Text>
      </TouchableOpacity>

      {/* Превью выбранного изображения */}
      {selectedImage && (
        <View style={styles.imageContainer}>
          <Image source={{ uri: selectedImage }} style={styles.image} />
        </View>
      )}

      {/* Кнопка поиска */}
      <TouchableOpacity
        style={[styles.button, styles.searchButton, (!selectedImage || !modelReady || !searchReady) && styles.buttonDisabled]}
        onPress={searchSimilar}
        disabled={!selectedImage || !modelReady || !searchReady || loading}
      >
        {loading ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Найти похожие</Text>
        )}
      </TouchableOpacity>

      {/* Результаты поиска */}
      {results.length > 0 && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Результаты ({results.length}):</Text>
          <ScrollView style={styles.resultsList}>
            {results.map((result, index) => (
              <View key={result.index} style={styles.resultItem}>
                <Text style={styles.resultIndex}>#{index + 1}</Text>
                <View style={styles.resultContent}>
                  <Text style={styles.resultCaption}>{result.caption}</Text>
                  <Text style={styles.resultDistance}>
                    Расстояние: {result.distance.toFixed(4)}
                  </Text>
                </View>
              </View>
            ))}
          </ScrollView>
        </View>
      )}

      {/* Статус модели и поиска */}
      <Text style={styles.status}>
        Модель: {modelReady ? 'Готова' : 'Загрузка...'} | Поиск: {searchReady ? 'Готов' : 'Загрузка...'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  topKContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  label: {
    fontSize: 16,
    marginRight: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 5,
    padding: 8,
    width: 80,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 15,
  },
  searchButton: {
    backgroundColor: '#34C759',
    marginTop: 10,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  imageContainer: {
    alignItems: 'center',
    marginVertical: 15,
  },
  image: {
    width: 200,
    height: 200,
    borderRadius: 8,
    resizeMode: 'contain',
  },
  resultsContainer: {
    flex: 1,
    marginTop: 20,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  resultsList: {
    flex: 1,
  },
  resultItem: {
    flexDirection: 'row',
    padding: 10,
    marginBottom: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
  },
  resultIndex: {
    fontSize: 16,
    fontWeight: 'bold',
    marginRight: 10,
    color: '#007AFF',
  },
  resultContent: {
    flex: 1,
  },
  resultCaption: {
    fontSize: 14,
    marginBottom: 5,
  },
  resultDistance: {
    fontSize: 12,
    color: '#666',
  },
  status: {
    textAlign: 'center',
    marginTop: 10,
    fontSize: 12,
    color: '#666',
  },
  centerContent: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  initProgress: {
    marginTop: 20,
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
  initHint: {
    marginTop: 10,
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    paddingHorizontal: 40,
  },
});

