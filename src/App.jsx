import { useState, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useModel } from './hooks/useModel';
import DrawingCanvas from './components/DrawingCanvas';
import TrainingPanel from './components/TrainingPanel';
import NetworkDiagram from './components/NetworkDiagram';
import FeatureMapViewer from './components/FeatureMapViewer';
import FilterViewer from './components/FilterViewer';
import CAMViewer from './components/CAMViewer';
import ModelControls from './components/ModelControls';
import InfoTooltip from './components/ui/InfoTooltip';
import Spinner from './components/ui/Spinner';

function App() {
  const {
    model,
    isTraining,
    isPaused,
    trainingProgress,
    trainingHistory,
    modelSummary,
    dataLoaded,
    dataLoadProgress,
    useFullDataset,
    // Model metadata
    modelSource,
    modelName,
    trainedEpochs,
    lastValAccuracy,
    // Actions
    initModel,
    loadData,
    train,
    stopTraining,
    pauseTraining,
    predict,
    saveModel,
    loadModel,
    loadPretrained,
    getLayerOutputs
  } = useModel();
  
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [allProbabilities, setAllProbabilities] = useState([]);
  const [layerOutputs, setLayerOutputs] = useState({});
  const [currentInput, setCurrentInput] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState('untrained');
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [userTrainedModel, setUserTrainedModel] = useState(null); // Will store weights
  const [showSaveSuccess, setShowSaveSuccess] = useState(false);
  
  const inputTensorRef = useRef(null);
  const savedWeightsRef = useRef(null);
  
  const handleSaveTrainedModel = useCallback(async () => {
    if (!model) return;
    
    // Save current weights
    const weights = model.getWeights().map(w => w.clone());
    if (savedWeightsRef.current) {
      savedWeightsRef.current.forEach(w => w.dispose());
    }
    savedWeightsRef.current = weights;
    setUserTrainedModel(true); 
    setShowSaveSuccess(true);
    setTimeout(() => setShowSaveSuccess(false), 3000);
  }, [model]);

  const handleImageReady = useCallback(async (tensor) => {
    setIsPredicting(true);
    
    if (!model) {
      // Initialize model if not exists
      initModel(0.001);
      
      // Store tensor for later
      inputTensorRef.current?.dispose();
      inputTensorRef.current = tensor;
      setCurrentInput(tensor);
      setIsPredicting(false);
      return;
    }
    
    // Dispose previous tensor
    inputTensorRef.current?.dispose();
    inputTensorRef.current = tensor;
    setCurrentInput(tensor);
    
    try {
      // Make prediction
      const predictionTensor = predict(tensor);
      if (!predictionTensor) {
        setIsPredicting(false);
        return;
      }
      
      const probs = await predictionTensor.data();
      const probsArray = Array.from(probs);
      
      // Get predicted class
      const predictedClass = probsArray.indexOf(Math.max(...probsArray));
      const conf = probsArray[predictedClass];
      
      setPrediction(predictedClass);
      setConfidence(conf);
      setAllProbabilities(probsArray);
      
      // Get layer outputs for visualization
      const outputs = getLayerOutputs(tensor);
      setLayerOutputs(outputs);
      
      predictionTensor.dispose();
    } catch (e) {
      console.error('Prediction error:', e);
    } finally {
      setIsPredicting(false);
    }
  }, [model, predict, getLayerOutputs, initModel]);
  
  const handleInitModel = useCallback((learningRate) => {
    const newModel = initModel(learningRate);
    
    // If we have a pending input, make prediction
    if (inputTensorRef.current && newModel) {
      setTimeout(() => {
        handleImageReady(inputTensorRef.current);
      }, 100);
    }
    
    return newModel;
  }, [initModel, handleImageReady]);
  
  // Handle model type selection change
  const handleModelTypeChange = useCallback(async (modelType) => {
    setSelectedModelType(modelType);
    setIsLoadingModel(true);
    
    // Reset predictions when switching models
    setPrediction(null);
    setConfidence(null);
    setAllProbabilities([]);
    setLayerOutputs({});
    
    try {
      if (modelType === 'untrained') {
        // Create a new untrained model
        initModel(0.001);
      } else if (modelType === 'pretrained') {
        // Load the pre-trained model from public folder
        await loadPretrained();
      } else if (modelType === 'user-saved' && savedWeightsRef.current) {
        // Create a new model architecture and load saved weights
        const newModel = initModel(0.001);
        newModel.setWeights(savedWeightsRef.current.map(w => w.clone()));
      }
      // 'user' type means user will train themselves - don't auto-load
    } catch (error) {
      console.error('Failed to load model:', error);
    } finally {
      setIsLoadingModel(false);
    }
  }, [initModel, loadPretrained]);
  
  // Get model status display info
  const getModelStatusInfo = () => {
    if (!model) {
      return {
        status: 'No Model',
        color: 'gray',
        icon: null
      };
    }
    
    if (modelSource === 'pretrained') {
      return {
        status: 'Pre-trained',
        name: 'MNIST CNN',
        color: 'purple',
        icon: (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
          </svg>
        )
      };
    }
    
    if (selectedModelType === 'user-saved') {
      return {
        status: 'Saved Model',
        name: 'User Trained',
        color: 'green',
        icon: (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        )
      };
    }
    
    if (modelSource === 'loaded') {
      return {
        status: 'Loaded',
        name: modelName,
        color: 'blue',
        icon: (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
        )
      };
    }
    
    return {
      status: trainedEpochs > 0 ? 'Trained' : 'Untrained',
      color: trainedEpochs > 0 ? 'green' : 'yellow',
      icon: trainedEpochs > 0 ? (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ) : (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
        </svg>
      )
    };
  };
  
  const modelStatus = getModelStatusInfo();
  
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-xl font-bold">CNN Visualizer</h1>
              <p className="text-xs text-gray-400">Train & visualize neural networks in your browser</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Model Status Badge - PROMINENT */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
              modelStatus.color === 'green' ? 'bg-green-900/50 border-green-700 text-green-400' :
              modelStatus.color === 'blue' ? 'bg-blue-900/50 border-blue-700 text-blue-400' :
              modelStatus.color === 'purple' ? 'bg-purple-900/50 border-purple-700 text-purple-400' :
              modelStatus.color === 'yellow' ? 'bg-yellow-900/50 border-yellow-700 text-yellow-400' :
              'bg-gray-700/50 border-gray-600 text-gray-400'
            }`}>
              <span className={`w-2 h-2 rounded-full ${
                modelStatus.color === 'green' ? 'bg-green-500' :
                modelStatus.color === 'blue' ? 'bg-blue-500' :
                modelStatus.color === 'purple' ? 'bg-purple-500' :
                modelStatus.color === 'yellow' ? 'bg-yellow-500' :
                'bg-gray-500'
              } ${isTraining ? 'animate-pulse' : ''}`} />
              
              {modelStatus.icon}
              
              <div className="text-sm">
                <span className="font-medium">{modelStatus.status}</span>
                {modelStatus.name && (
                  <span className="text-gray-400 ml-1">({modelStatus.name})</span>
                )}
              </div>
              
              {trainedEpochs > 0 && (
                <span className="text-xs opacity-75">
                  {trainedEpochs} epochs
                </span>
              )}
              
              {lastValAccuracy !== null && (
                <span className="text-xs bg-black/30 px-1.5 py-0.5 rounded">
                  {(lastValAccuracy * 100).toFixed(1)}% acc
                </span>
              )}
            </div>
            
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-white transition-colors cursor-pointer"
            >
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Drawing & Prediction */}
          <div className="space-y-6">
            {/* Model Selector Section */}
            <div className="bg-gray-800 rounded-xl p-4 border border-blue-900/30">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  Model Selection
                </h2>
                {isLoadingModel && <Spinner size="sm" />}
              </div>
              
              <div className="space-y-4">
                <div className="flex flex-col gap-2">
                  <label className="text-sm text-gray-400">Choose Active Model:</label>
                  <select
                    value={selectedModelType}
                    onChange={(e) => handleModelTypeChange(e.target.value)}
                    disabled={isTraining || isLoadingModel}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <option value="untrained">Untrained (Random Weights)</option>
                    <option value="pretrained">Pre-trained MNIST (99% Acc)</option>
                    {/* <option value="user">Current Workspace Model</option> */}
                    {userTrainedModel && (
                      <option value="user-saved">My Saved Model</option>
                    )}
                  </select>
                </div>

                {trainedEpochs > 0 && !userTrainedModel && !isTraining && (
                  <button
                    onClick={handleSaveTrainedModel}
                    className="w-full py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Save as Active Model
                  </button>
                )}

                {showSaveSuccess && (
                  <div className="text-xs text-green-400 text-center animate-fade-in">
                    Model saved to selection list!
                  </div>
                )}
              </div>
            </div>

            {/* Drawing Canvas */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                  Draw a Digit
                </h2>
                <InfoTooltip 
                  text="Draw a handwritten digit (0-9) using your mouse or touch. The model will try to recognize it. You can also upload an image file."
                  position="left"
                />
              </div>
              <DrawingCanvas 
                onImageReady={handleImageReady} 
                disabled={isTraining}
                isPredicting={isPredicting}
              />
            </div>
            
            {/* Prediction Result */}
            {prediction !== null && (
              <div className="bg-gray-800 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Prediction
                  </h2>
                  <InfoTooltip 
                    text="Shows the model's prediction with confidence scores for each digit class (0-9). Higher bars indicate higher confidence."
                    position="left"
                  />
                </div>
                
                {/* Model used indicator */}
                <div className="mb-3 text-xs text-gray-500 flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Using: {modelSource === 'pretrained' ? 'Pre-trained MNIST CNN' : modelSource === 'loaded' ? modelName || 'Loaded Model' : trainedEpochs > 0 ? `Trained Model (${trainedEpochs} epochs)` : 'Untrained Model'}
                </div>
                
                <div className="flex items-center gap-4 mb-4">
                  <div className="text-6xl font-bold text-blue-400">{prediction}</div>
                  <div className="flex-1">
                    <div className="text-sm text-gray-400">Confidence</div>
                    <div className="text-2xl font-semibold text-white">
                      {(confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                
                {/* Probability bars */}
                <div className="space-y-1">
                  {allProbabilities.map((prob, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="text-xs text-gray-400 w-4">{idx}</span>
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-300 ${
                            idx === prediction ? 'bg-blue-500' : 'bg-gray-500'
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-12 text-right">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Model Controls */}
            <ModelControls
              model={model}
              onSave={saveModel}
              onLoad={loadModel}
              modelSource={modelSource}
              modelName={modelName}
            />
          </div>
          
          {/* Middle Column - Training & Network */}
          <div className="space-y-6">
            <TrainingPanel
              model={model}
              isTraining={isTraining}
              isPaused={isPaused}
              trainingProgress={trainingProgress}
              trainingHistory={trainingHistory}
              dataLoaded={dataLoaded}
              dataLoadProgress={dataLoadProgress}
              useFullDataset={useFullDataset}
              onLoadData={loadData}
              onTrain={train}
              onStop={stopTraining}
              onPause={pauseTraining}
              onInitModel={handleInitModel}
            />
            
            <NetworkDiagram modelSummary={modelSummary} />
          </div>
          
          {/* Right Column - Visualizations */}
          <div className="space-y-6">
            <FeatureMapViewer 
              layerOutputs={layerOutputs} 
              title="Feature Maps / Activations"
            />
            
            <FilterViewer model={model} />
            
            <CAMViewer
              model={model}
              inputTensor={currentInput}
              prediction={prediction}
            />
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 px-6 py-4 mt-8">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-sm text-gray-400">
          <div>
            Built with TensorFlow.js | Runs entirely in your browser
          </div>
          <div className="flex items-center gap-4">
            <span>Memory: {(tf.memory().numBytes / 1024 / 1024).toFixed(1)} MB</span>
            <span>Tensors: {tf.memory().numTensors}</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
