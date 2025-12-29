/**
 * Image Search App with ONNX Runtime Mobile
 * 
 * @format
 */

import React, { useEffect, useState } from 'react';
import { StatusBar, StyleSheet, useColorScheme } from 'react-native';
import {
  SafeAreaProvider,
} from 'react-native-safe-area-context';
import ImageSearchScreen from './src/components/ImageSearchScreen';
import { prepareModelAssets } from './src/services/onnxService';


function App() {
  const isDarkMode = useColorScheme() === 'dark';
  const [assetsReady, setAssetsReady] = useState(false);

  useEffect(() => {
    async function initAssets() {
      await prepareModelAssets();
      setAssetsReady(true);
    }
    initAssets();
  }, []);

  if (!assetsReady) {
    return null; // loader or spinner
  }

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <ImageSearchScreen />
    </SafeAreaProvider>
  );
}

export default App;
