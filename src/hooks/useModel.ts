import { useState, useCallback, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { createModel, compileModel, getModelSummary, downloadModel, loadModelFromFiles, loadPretrainedModel, LayerSummary } from '../utils/modelUtils';
import { loadSampleMNIST, loadFullMNIST, getBatch, MNISTData } from '../utils/mnistLoader';

export interface TrainingProgress {
  epoch: number;
  batch: number;
  totalBatches?: number;
  loss: number;
  accuracy: number;
}

export interface TrainingHistory {
  loss: number[];
  accuracy: number[];
  valLoss: number[];
  valAccuracy: number[];
}

export interface LayerOutput {
  tensor: tf.Tensor;
  shape: number[];
  type: string;
  index: number;
}

export type ModelSource = 'new' | 'loaded' | 'pretrained' | null;

export function useModel() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({ epoch: 0, batch: 0, loss: 0, accuracy: 0 });
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistory>({ loss: [], accuracy: [], valLoss: [], valAccuracy: [] });
  const [modelSummary, setModelSummary] = useState<LayerSummary[]>([]);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [dataLoadProgress, setDataLoadProgress] = useState(0);
  const [useFullDataset, setUseFullDataset] = useState(false);
  
  // Model metadata for tracking source and status
  const [modelSource, setModelSource] = useState<ModelSource>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  const [trainedEpochs, setTrainedEpochs] = useState(0); // cumulative epochs
  const [lastValAccuracy, setLastValAccuracy] = useState<number | null>(null); // last validation accuracy
  
  const dataRef = useRef<MNISTData | null>(null);
  const stopTrainingRef = useRef(false);
  const pauseTrainingRef = useRef(false);
  
  // Initialize model
  const initModel = useCallback((learningRate = 0.001) => {
    const newModel = createModel();
    compileModel(newModel, learningRate);
    setModel(newModel);
    setModelSummary(getModelSummary(newModel));
    setTrainingHistory({ loss: [], accuracy: [], valLoss: [], valAccuracy: [] });
    setModelSource('new');
    setModelName(null);
    setTrainedEpochs(0);
    setLastValAccuracy(null);
    return newModel;
  }, []);
  
  // Load data
  const loadData = useCallback(async (useFull = false) => {
    setDataLoaded(false);
    setDataLoadProgress(0);
    
    try {
      let data: MNISTData;
      if (useFull) {
        data = await loadFullMNIST((progress) => setDataLoadProgress(progress));
      } else {
        data = await loadSampleMNIST();
        setDataLoadProgress(1);
      }
      
      dataRef.current = data;
      setDataLoaded(true);
      setUseFullDataset(useFull);
      
      return data;
    } catch (error) {
      console.error('Failed to load data:', error);
      throw error;
    }
  }, []);
  
  // Train model
  const train = useCallback(async (options: {
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    validationSplit?: number;
    onEpochEnd?: (metrics: any) => void;
    onBatchEnd?: (metrics: any) => void;
    modelOverride?: tf.LayersModel | null;
  } = {}) => {
    const {
      epochs = 10,
      batchSize = 32,
      learningRate = 0.001,
      onEpochEnd,
      onBatchEnd,
      modelOverride = null
    } = options;
    
    const activeModel = modelOverride || model;
    
    if (!activeModel || !dataRef.current) {
      console.error('Model or data not loaded');
      return;
    }
    
    // Recompile with new learning rate
    compileModel(activeModel, learningRate);
    
    setIsTraining(true);
    setIsPaused(false);
    stopTrainingRef.current = false;
    pauseTrainingRef.current = false;
    
    const { trainImages, trainLabels, testImages, testLabels, numTrain } = dataRef.current;
    const numBatches = Math.ceil(numTrain / batchSize);
    
    const history: TrainingHistory = { loss: [], accuracy: [], valLoss: [], valAccuracy: [] };
    let completedEpochs = 0;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (stopTrainingRef.current) break;
      
      // Wait while paused
      while (pauseTrainingRef.current && !stopTrainingRef.current) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (stopTrainingRef.current) break;
      
      let epochLoss = 0;
      let epochAcc = 0;
      let batchCount = 0;
      
      for (let batch = 0; batch < numBatches; batch++) {
        if (stopTrainingRef.current) break;
        
        while (pauseTrainingRef.current && !stopTrainingRef.current) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (stopTrainingRef.current) break;
        
        const { images, labels } = getBatch(trainImages, trainLabels, batchSize, batch);
        
        const result = await activeModel.trainOnBatch(images, labels);
        const batchLoss = Array.isArray(result) ? result[0] : result;
        const batchAcc = Array.isArray(result) ? result[1] : 0;
        
        epochLoss += batchLoss;
        epochAcc += batchAcc;
        batchCount++;
        
        // Cleanup batch tensors
        images.dispose();
        labels.dispose();
        
        setTrainingProgress({
          epoch: epoch + 1,
          batch: batch + 1,
          totalBatches: numBatches,
          loss: batchLoss,
          accuracy: batchAcc
        });
        
        if (onBatchEnd) {
          onBatchEnd({ epoch, batch, loss: batchLoss, accuracy: batchAcc });
        }
        
        // Small delay to allow UI updates
        if (batch % 10 === 0) {
          await tf.nextFrame();
        }
      }
      
      if (stopTrainingRef.current) break;
      
      // Calculate validation metrics
      const valResult = activeModel.evaluate(testImages, testLabels) as tf.Scalar[];
      const valLoss = (await valResult[0].data())[0];
      const valAcc = (await valResult[1].data())[0];
      valResult[0].dispose();
      valResult[1].dispose();
      
      const avgLoss = epochLoss / batchCount;
      const avgAcc = epochAcc / batchCount;
      
      history.loss.push(avgLoss);
      history.accuracy.push(avgAcc);
      history.valLoss.push(valLoss);
      history.valAccuracy.push(valAcc);
      
      setTrainingHistory({ ...history });
      setLastValAccuracy(valAcc);
      
      completedEpochs++;
      
      if (onEpochEnd) {
        onEpochEnd({
          epoch: epoch + 1,
          loss: avgLoss,
          accuracy: avgAcc,
          valLoss,
          valAccuracy: valAcc
        });
      }
      
      await tf.nextFrame();
    }
    
    // Update total trained epochs
    setTrainedEpochs(prev => prev + completedEpochs);
    
    setIsTraining(false);
    setIsPaused(false);
    
    return history;
  }, [model]);
  
  // Stop training
  const stopTraining = useCallback(() => {
    stopTrainingRef.current = true;
    pauseTrainingRef.current = false;
  }, []);
  
  // Pause/resume training
  const pauseTraining = useCallback(() => {
    pauseTrainingRef.current = !pauseTrainingRef.current;
    setIsPaused(pauseTrainingRef.current);
  }, []);
  
  // Predict
  const predict = useCallback((inputTensor: tf.Tensor) => {
    if (!model) return null;
    
    return tf.tidy(() => {
      const prediction = model.predict(inputTensor) as tf.Tensor;
      return prediction;
    });
  }, [model]);
  
  // Download model
  const saveModel = useCallback(async (name = 'mnist-cnn') => {
    if (!model) return;
    await downloadModel(model, name);
  }, [model]);
  
  // Load model from files
  const loadModel = useCallback(async (jsonFile: File, weightsFiles: File[]) => {
    try {
      const loadedModel = await loadModelFromFiles(jsonFile, weightsFiles);
      compileModel(loadedModel);
      setModel(loadedModel);
      setModelSummary(getModelSummary(loadedModel));
      
      // Update metadata
      setModelSource('loaded');
      setModelName(jsonFile.name.replace('.json', ''));
      setTrainedEpochs(0); // Unknown, but can continue training
      setLastValAccuracy(null);
      setTrainingHistory({ loss: [], accuracy: [], valLoss: [], valAccuracy: [] });
      
      return loadedModel;
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }, []);
  
  // Load pre-trained model from public folder
  const loadPretrained = useCallback(async () => {
    try {
      const loadedModel = await loadPretrainedModel();
      compileModel(loadedModel);
      setModel(loadedModel);
      setModelSummary(getModelSummary(loadedModel));
      
      // Update metadata
      setModelSource('pretrained');
      setModelName('Pre-trained MNIST CNN');
      setTrainedEpochs(0); // Pre-trained, epochs unknown
      setLastValAccuracy(null);
      setTrainingHistory({ loss: [], accuracy: [], valLoss: [], valAccuracy: [] });
      
      return loadedModel;
    } catch (error) {
      console.error('Failed to load pre-trained model:', error);
      throw error;
    }
  }, []);
  
  // Get intermediate layer outputs for visualization
  const getLayerOutputs = useCallback((inputTensor: tf.Tensor) => {
    if (!model) return {};
    
    const outputs: Record<string, LayerOutput> = {};
    
    model.layers.forEach((layer, index) => {
      const layerType = layer.getClassName();
      if (layerType === 'Conv2D' || layerType === 'MaxPooling2D' || layerType === 'Flatten' || layerType === 'Dense') {
        try {
          const intermediateModel = tf.model({
            inputs: model.inputs,
            outputs: layer.output as tf.SymbolicTensor
          });
          
          const output = intermediateModel.predict(inputTensor) as tf.Tensor;
          outputs[layer.name] = {
            tensor: output,
            shape: output.shape,
            type: layerType,
            index
          };
        } catch (e) {
          console.warn(`Failed to get output for layer ${layer.name}:`, e);
        }
      }
    });
    
    return outputs;
  }, [model]);
  
  // Get filter weights
  const getFilterWeights = useCallback((layerName: string) => {
    if (!model) return null;
    
    try {
      const layer = model.getLayer(layerName);
      if (layer.getClassName() !== 'Conv2D') return null;
      
      const weights = layer.getWeights()[0];
      return weights;
    } catch (e) {
      console.warn(`Failed to get weights for layer ${layerName}:`, e);
      return null;
    }
  }, [model]);
  
  // Cleanup model when it changes (dispose old model)
  useEffect(() => {
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [model]);

  // Cleanup data only on unmount
  useEffect(() => {
    return () => {
      if (dataRef.current) {
        dataRef.current.trainImages?.dispose();
        dataRef.current.trainLabels?.dispose();
        dataRef.current.testImages?.dispose();
        dataRef.current.testLabels?.dispose();
      }
    };
  }, []);
  
  return {
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
    getLayerOutputs,
    getFilterWeights
  };
}
