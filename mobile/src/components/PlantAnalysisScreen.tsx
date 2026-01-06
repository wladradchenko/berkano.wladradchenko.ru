import React, { useState, useEffect } from 'react';
import Svg, { Path, Circle, Rect, G } from 'react-native-svg';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  FlatList,
  ImageBackground
} from 'react-native';
import { useTranslation } from 'react-i18next';
import { launchImageLibrary, ImagePickerResponse, Asset } from 'react-native-image-picker';
import { initializeModelManager, isModelLoaded } from '../services/modelManager';
import { initializeSearch } from '../services/embeddingsService';
import { classifyAge, classifyPlant, AgeClassificationResult, PlantClassificationResult } from '../services/plantClassificationService';
import { searchDisease, DiseaseSearchResult } from '../services/diseaseSearchService';
import { preprocessImageForAge, preprocessImageForPlant } from '../services/imagePreprocessingPlant';
import { preprocessImage } from '../services/imagePreprocessing';
import { BlurView } from '@react-native-community/blur';

export default function PlantAnalysisScreen() {
  const { t, i18n } = useTranslation();
  
  // Состояние инициализации
  const [initializing, setInitializing] = useState(true);
  const [initProgress, setInitProgress] = useState('');
  
  // Блок 1: Классификация растения
  const [plantImage, setPlantImage] = useState<string | null>(null);
  const [plantResults, setPlantResults] = useState<PlantClassificationResult[]>([]);
  const [ageResults, setAgeResults] = useState<AgeClassificationResult[]>([]);
  
  // Блок 2: Поиск болезней
  const [diseaseImages, setDiseaseImages] = useState<string[]>([]);
  const [diseaseResults, setDiseaseResults] = useState<DiseaseSearchResult[][]>([]);
  
  // Общее состояние
  const [analyzing, setAnalyzing] = useState(false);
  const [modelsReady, setModelsReady] = useState(false);

  const toggleLanguage = () => {
    const newLang = i18n.language === 'ru' ? 'en' : 'ru';
    i18n.changeLanguage(newLang);
  };

  useEffect(() => {
    const init = async () => {
      try {
        setInitProgress(t('initialization.modelManager'));
        await initializeModelManager();
        
        setInitProgress(t('initialization.loadingData'));
        await initializeSearch();
        
        setModelsReady(true);
        setInitProgress(t('initialization.ready'));
        setTimeout(() => {
          setInitializing(false);
        }, 1000);
      } catch (error: any) {
        console.error('Ошибка инициализации:', error);
        setInitializing(false);
        Alert.alert(
          t('errors.initFailed'),
          `${t('errors.initFailed')}: ${error?.message || error}\n\n${t('errors.tryAgain')}`,
        );
      }
    };
    init();
  }, []);

  const selectPlantImage = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 1,
        selectionLimit: 1,
      },
      (response: ImagePickerResponse) => {
        if (response.assets && response.assets[0]) {
          setPlantImage(response.assets[0].uri || null);
          setPlantResults([]);
        }
      },
    );
  };

  const selectDiseaseImages = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 1,
        selectionLimit: 10,
      },
      (response: ImagePickerResponse) => {
        if (response.assets && response.assets.length > 0) {
          const uris = response.assets
            .map(asset => asset.uri)
            .filter((uri): uri is string => uri !== null && uri !== undefined);
          setDiseaseImages(uris);
          setDiseaseResults([]);
        }
      },
    );
  };

  const removeDiseaseImage = (index: number) => {
    const newImages = [...diseaseImages];
    newImages.splice(index, 1);
    setDiseaseImages(newImages);
  };

  const analyze = async () => {
    if (!plantImage && diseaseImages.length === 0) {
      Alert.alert(t('errors.noImages'), t('errors.noImagesMessage'));
      return;
    }

    if (!modelsReady) {
      Alert.alert(t('errors.notReady'), t('errors.notReadyMessage'));
      return;
    }

    setAnalyzing(true);
    setPlantResults([]);
    setDiseaseResults([]);

    try {
      // Блок 1: Классификация растения
      if (plantImage) {
        console.log('Classifying plant...');
        const preprocessed = await preprocessImageForPlant(plantImage);
        const results = await classifyPlant(preprocessed.data, 5);
        setPlantResults(results);
        console.log('Plant classification completed');

        console.log('Classifying age...');
        const preprocessedAge = await preprocessImageForAge(plantImage);
        const resultsAge = await classifyAge(preprocessedAge.data);
        setAgeResults(resultsAge);
        console.log('Age classification completed');
      }

      // Блок 2: Поиск болезней
      if (diseaseImages.length > 0) {
        console.log(`Analyzing ${diseaseImages.length} disease images...`);
        const allResults: DiseaseSearchResult[][] = [];

        for (const imageUri of diseaseImages) {
          const preprocessed = await preprocessImage(imageUri);
          const results = await searchDisease(preprocessed.data, 10);
          allResults.push(results);
        }

        setDiseaseResults(allResults);
        console.log('Disease search completed');
      }
    } catch (error: any) {
      console.error('Ошибка при анализе:', error);
      Alert.alert(t('errors.analysisFailed'), `${t('errors.analysisFailed')}: ${error?.message || error}`);
    } finally {
      setAnalyzing(false);
    }
  };

  if (initializing) {
    return (
      <View style={[styles.container, styles.centerContent]}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.initProgress}>{initProgress}</Text>
        <Text style={styles.initHint}>
          {t('initialization.hint')}
        </Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={{marginBottom: 10, marginTop: 50, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'}}>
        <Text style={{fontSize: 16}}>Berkano</Text>
        <View style={{ flexDirection: 'row', gap: 12 }}>
          <TouchableOpacity onPress={() =>{}}>
            <Svg width={24} height={24} viewBox="0 0 24 24"><Path d="M15.7955 15.8111L21 21M18 10.5C18 14.6421 14.6421 18 10.5 18C6.35786 18 3 14.6421 3 10.5C3 6.35786 6.35786 3 10.5 3C14.6421 3 18 6.35786 18 10.5Z" stroke="#000000" fill="transparent" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" /></Svg>
          </TouchableOpacity>
          <TouchableOpacity onPress={toggleLanguage}>
            <Svg width={24} height={24} viewBox="0 0 24 24" stroke="#000000" strokeWidth="1.1" strokeLinecap="square" strokeLinejoin="miter" fill="none">
              <Path strokeLinecap="round" d="M12,22 C14.6666667,19.5757576 16,16.2424242 16,12 C16,7.75757576 14.6666667,4.42424242 12,2 C9.33333333,4.42424242 8,7.75757576 8,12 C8,16.2424242 9.33333333,19.5757576 12,22 Z" />
              <Path strokeLinecap="round" d="M2.5 9L21.5 9M2.5 15L21.5 15" />
              <Circle cx="12" cy="12" r="10" fill="none" />
            </Svg>
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.header}>
        <Text style={styles.title}>{t('app.title')}</Text>
      </View>

      <View style={{ marginBottom: 20, flexDirection: 'row', alignItems: 'center', gap: 12 }}>
        <TouchableOpacity style={{ flexDirection: 'row', alignItems: 'center', gap: 5, borderWidth: 1, borderColor: '#b7c2b0', borderRadius: 999, padding: 10, paddingHorizontal: 15 }} onPress={() =>{}}>
          <Svg width={24} height={24} viewBox="0 0 32 32" fill="none">
            <Path fill="#000000" d="M18,28H14a2,2,0,0,1-2-2V18.41L4.59,11A2,2,0,0,1,4,9.59V6A2,2,0,0,1,6,4H26a2,2,0,0,1,2,2V9.59A2,2,0,0,1,27.41,11L20,18.41V26A2,2,0,0,1,18,28ZM6,6V9.59l8,8V26h4V17.59l8-8V6Z" />
          </Svg>
          <Text style={{ fontSize: 16 }}>Filters</Text>
        </TouchableOpacity>
      </View>

      {/* Блок 1: Классификация растения */}
      <View style={styles.block}>
        <Text style={styles.blockTitle}>{t('plantClassification.title')}</Text>
        <Text style={styles.blockDescription}>{t('plantClassification.description')}</Text>

        {plantImage ? (
          <ImageBackground source={{ uri: plantImage }} style={{ width: '100%', height: 220, borderRadius: 20, overflow: 'hidden', justifyContent: 'flex-end' }} imageStyle={{ borderRadius: 20 }}>
            <BlurView blurType="light" blurAmount={7} reducedTransparencyFallbackColor="white" style={{ backgroundColor: 'rgba(215, 222, 207,0.15)', padding: 15, flexDirection: 'column', }}>
              <Text style={{ color: '#000', fontSize: 20, fontWeight: 'bold', marginBottom: 5, paddingHorizontal: 10 }}>{plantResults[0]?.speciesName || ''} {ageResults[0]?.age || ''} years old with {ageResults[0]?.leafCount || ''} leaves</Text>
              {plantResults[0] && (
                <View style={{ height: 6, backgroundColor: '#fff', borderRadius: 3, overflow: 'hidden' }}>
                  <View style={{ width: `${plantResults[0].probability}%`, height: 6, backgroundColor: '#00FF7F' }} />
                </View>
              )}
            </BlurView>
            <TouchableOpacity style={styles.removeButton} onPress={() => {setPlantImage(null)}}>
              <Text style={styles.removeButtonText}>{t('plantClassification.remove')}</Text>
            </TouchableOpacity>
          </ImageBackground>
        ) : (
          <TouchableOpacity style={{ backgroundColor: '#242424', borderRadius: 20, padding: 18, alignItems: 'center' }} onPress={selectPlantImage}>
            <Text style={{ color: '#fff', fontSize: 16, fontWeight: 'bold' }}>{t('plantClassification.selectButton')}</Text>
          </TouchableOpacity>
        )}

        {plantResults.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: 15 }}>
            {plantResults.map((result, idx) => (
              <View key={idx} style={{ backgroundColor: '#fff', borderRadius: 15, padding: 10, marginRight: 10, width: 150 }}>
                <Text style={styles.resultSpecies}>{result.speciesName}</Text>
                <View style={{ backgroundColor: '#eee', height: 6, borderRadius: 3, marginBottom: 5 }}>
                  <View style={{ width: `${result.probability}%`, height: 6, backgroundColor: '#007AFF', borderRadius: 3 }} />
                </View>
                <Text style={styles.resultProbability}>{result.probability.toFixed(2)}%</Text>
              </View>
            ))}
          </ScrollView>
        )}
      </View>

      {/* Блок 2: Поиск болезней */}
      <View style={styles.block}>
        <Text style={styles.blockTitle}>{t('diseaseSearch.title')}</Text>
        <Text style={styles.blockDescription}>
          {t('diseaseSearch.description')}
        </Text>

        <TouchableOpacity style={styles.button} onPress={selectDiseaseImages}>
          <Text style={styles.buttonText}>
            {diseaseImages.length > 0
              ? `${t('diseaseSearch.changeButton')} (${diseaseImages.length}/10)`
              : t('diseaseSearch.selectButton')}
          </Text>
        </TouchableOpacity>

        {diseaseImages.length > 0 && (
          <View style={styles.imagesGrid}>
            {diseaseImages.map((uri, index) => (
              <View key={index} style={styles.imageItem}>
                <Image source={{ uri }} style={styles.smallImage} />
                <TouchableOpacity
                  style={styles.removeSmallButton}
                  onPress={() => removeDiseaseImage(index)}
                >
                  <Text style={styles.removeSmallButtonText}>×</Text>
                </TouchableOpacity>
                {diseaseResults[index] && (
                  <View style={styles.diseaseResultsBadge}>
                    <Text style={styles.diseaseResultsText}>
                      {diseaseResults[index].length} {t('diseaseSearch.resultsCount')}
                    </Text>
                  </View>
                )}
              </View>
            ))}
          </View>
        )}

        {diseaseResults.length > 0 && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>{t('diseaseSearch.resultsTitle')}</Text>
            {diseaseResults.map((results, imageIndex) => (
              <View key={imageIndex} style={styles.diseaseImageResults}>
                <Text style={styles.diseaseImageTitle}>
                  {t('diseaseSearch.photo')} #{imageIndex + 1}:
                </Text>
                {results.slice(0, 3).map((result, resultIndex) => (
                  <View key={resultIndex} style={styles.resultItem}>
                    <Text style={styles.resultIndex}>#{resultIndex + 1}</Text>
                    <View style={styles.resultContent}>
                      <Text style={styles.resultCaption}>{result.caption}</Text>
                      <Text style={styles.resultDistance}>
                        {t('diseaseSearch.distance')}: {result.distance.toFixed(4)}
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Кнопка анализа */}
      <TouchableOpacity
        style={[styles.analyzeButton, (!plantImage && diseaseImages.length === 0) && styles.buttonDisabled]}
        onPress={analyze}
        disabled={analyzing || (!plantImage && diseaseImages.length === 0) || !modelsReady}
      >
        {analyzing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.analyzeButtonText}>{t('app.analyze')}</Text>
        )}
      </TouchableOpacity>

      {/* Статус */}
      <Text style={styles.status}>
        {t('app.status.models')}: {modelsReady ? t('app.status.ready') : t('app.status.loading')}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#edf1e8',
    color: "#252525",
  },
  centerContent: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    marginTop: 20,
  },
  title: {
    fontSize: 38,
    flex: 1,
  },
  languageButton: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    backgroundColor: '#007AFF',
    borderRadius: 5,
    marginLeft: 10,
  },
  languageButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  block: {
    marginBottom: 30,
    padding: 15,
    backgroundColor: '#d7decf',
    borderRadius: 25,
  },
  blockTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  blockDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
  button: {
    backgroundColor: '#242424',
    padding: 20,
    borderRadius: 99,
    alignItems: 'center',
    marginBottom: 15,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'regular',
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
  smallImage: {
    width: 100,
    height: 100,
    borderRadius: 8,
    resizeMode: 'cover',
  },
  removeButton: {
    padding: 8,
    backgroundColor: '#ff3b30',
    borderRadius: 5,
    width: '100%',
    height: 40,
  },
  removeButtonText: {
    color: '#fff',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  imagesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 10,
  },
  imageItem: {
    margin: 5,
    position: 'relative',
  },
  removeSmallButton: {
    position: 'absolute',
    top: -5,
    right: -5,
    backgroundColor: '#ff3b30',
    borderRadius: 12,
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeSmallButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  diseaseResultsBadge: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 4,
    borderBottomLeftRadius: 8,
    borderBottomRightRadius: 8,
  },
  diseaseResultsText: {
    color: '#fff',
    fontSize: 10,
    textAlign: 'center',
  },
  resultsContainer: {
    marginTop: 15,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  resultItem: {
    flexDirection: 'row',
    padding: 10,
    marginBottom: 10,
    backgroundColor: '#fff',
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
  resultSpecies: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  resultCaption: {
    fontSize: 14,
    marginBottom: 5,
  },
  resultProbability: {
    fontSize: 12,
    color: '#666',
  },
  resultDistance: {
    fontSize: 12,
    color: '#666',
  },
  diseaseImageResults: {
    marginBottom: 20,
    padding: 10,
    backgroundColor: '#fff',
    borderRadius: 8,
  },
  diseaseImageTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  analyzeButton: {
    backgroundColor: '#242424',
    padding: 20,
    borderRadius: 99,
    alignItems: 'center',
    marginBottom: 15,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  status: {
    textAlign: 'center',
    marginTop: 10,
    fontSize: 12,
    color: '#666',
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

