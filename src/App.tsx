import { useState, useCallback, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { useModel, LayerOutput } from './hooks/useModel';
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

  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [allProbabilities, setAllProbabilities] = useState<number[]>([]);
  const [layerOutputs, setLayerOutputs] = useState<Record<string, LayerOutput>>({});
  const [currentInput, setCurrentInput] = useState<tf.Tensor | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState('untrained');
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [userTrainedModel, setUserTrainedModel] = useState<boolean | null>(null); // Will store weights
  const [showSaveSuccess, setShowSaveSuccess] = useState(false);

  const inputTensorRef = useRef<tf.Tensor | null>(null);
  const savedWeightsRef = useRef<tf.Tensor[] | null>(null);

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

  const handleImageReady = useCallback(async (tensor: tf.Tensor) => {
    setIsPredicting(true);

    let activeModel = model;
    if (!activeModel) {
      // Initialize model if not exists
      activeModel = initModel(0.001);
    }

    // Dispose previous tensor
    inputTensorRef.current?.dispose();
    inputTensorRef.current = tensor;
    setCurrentInput(tensor);

    try {
      // Make prediction
      const predictionTensor = predict(tensor) || (activeModel ? tf.tidy(() => (activeModel as tf.LayersModel).predict(tensor) as tf.Tensor) : null);
      if (!predictionTensor) {
        setIsPredicting(false);
        return;
      }

      const probs = await predictionTensor.data();
      const probsArray = Array.from(probs);

      // Get predicted class
      const predictedClass = probsArray.indexOf(Math.max(...probsArray));
      const conf = probsArray[predictedClass] as number;

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

  const handleInitModel = useCallback((learningRate: number) => {
    const newModel = initModel(learningRate);

    // If we have a pending input, make prediction
    if (inputTensorRef.current && newModel) {
      setTimeout(() => {
        if (inputTensorRef.current) {
          handleImageReady(inputTensorRef.current);
        }
      }, 100);
    }

    return newModel;
  }, [initModel, handleImageReady]);

  // Handle model type selection change
  const handleModelTypeChange = useCallback(async (modelType: string) => {
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

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-xl font-bold">CNN Visualizer</h1>
              <p className="text-xs text-gray-400">Train & visualize neural networks in your browser</p>
            </div>
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
              </div>
            </div>

            {/* Drawing Canvas */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
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
                    Prediction
                  </h2>
                  <InfoTooltip
                    text="Shows the model's prediction with confidence scores for each digit class (0-9). Higher bars indicate higher confidence."
                    position="left"
                  />
                </div>

                {/* Model used indicator */}
                <div className="mb-3 text-xs text-gray-500 flex items-center gap-1">
                  Using: {
                    selectedModelType === 'user-saved' ? 'My Saved Model' :
                      modelSource === 'pretrained' ? 'Pre-trained MNIST CNN' :
                        modelSource === 'loaded' ? modelName || 'Loaded Model' :
                          trainedEpochs > 0 ? `Trained Model (${trainedEpochs} epochs)` :
                            'Untrained Model'
                  }
                </div>

                <div className="flex items-center gap-4 mb-4">
                  <div className="text-6xl font-bold text-blue-400">{prediction}</div>
                  <div className="flex-1">
                    <div className="text-sm text-gray-400">Confidence</div>
                    <div className="text-2xl font-semibold text-white">
                      {((confidence || 0) * 100).toFixed(1)}%
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
                          className={`h-2 rounded-full transition-all duration-300 ${idx === prediction ? 'bg-blue-500' : 'bg-gray-500'
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

          {/* Middle Column - Visualizations */}
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

          {/* Right Column - Training & Network */}
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
              // New props for relocated save button
              onSaveTrainedModel={handleSaveTrainedModel}
              userTrainedModel={userTrainedModel}
              trainedEpochs={trainedEpochs}
              showSaveSuccess={showSaveSuccess}
            />

            <NetworkDiagram modelSummary={modelSummary} />
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
