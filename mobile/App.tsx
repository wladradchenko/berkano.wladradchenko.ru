/**
 * Image Search App with ONNX Runtime Mobile
 * 
 * @format
 */

import { StatusBar, useColorScheme } from 'react-native';
import {
  SafeAreaProvider,
} from 'react-native-safe-area-context';
import PlantAnalysisScreen from './src/components/PlantAnalysisScreen';

function App() {
  const isDarkMode = useColorScheme() === 'dark';

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />
      <PlantAnalysisScreen />
    </SafeAreaProvider>
  );
}

export default App;
