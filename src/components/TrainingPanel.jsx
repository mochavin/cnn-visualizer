import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Spinner from './ui/Spinner';
import InfoTooltip from './ui/InfoTooltip';
import DisabledTooltip from './ui/DisabledTooltip';

export default function TrainingPanel({
  model,
  isTraining,
  isPaused,
  trainingProgress,
  trainingHistory,
  dataLoaded,
  dataLoadProgress,
  useFullDataset,
  onLoadData,
  onTrain,
  onStop,
  onPause,
  onInitModel
}) {
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [dataSource, setDataSource] = useState('sample');
  const [isLoadingData, setIsLoadingData] = useState(false);
  
  // Get disabled reasons
  const getLoadDataDisabledReason = () => {
    if (isTraining) return 'Training in progress';
    if (dataLoaded) return 'Data already loaded';
    if (isLoadingData) return 'Loading data...';
    return null;
  };
  
  const getTrainDisabledReason = () => {
    if (!dataLoaded) return 'Load data first';
    return null;
  };
  
  const loadDataDisabledReason = getLoadDataDisabledReason();
  const trainDisabledReason = getTrainDisabledReason();
  
  const handleLoadData = async () => {
    setIsLoadingData(true);
    try {
      await onLoadData(dataSource === 'full');
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setIsLoadingData(false);
    }
  };
  
  const handleStartTraining = async () => {
    if (!model) {
      onInitModel(learningRate);
    }
    
    await onTrain({
      epochs,
      batchSize,
      learningRate
    });
  };
  
  // Prepare chart data
  const chartData = trainingHistory.loss.map((loss, index) => ({
    epoch: index + 1,
    loss: loss.toFixed(4),
    accuracy: (trainingHistory.accuracy[index] * 100).toFixed(1),
    valLoss: trainingHistory.valLoss[index]?.toFixed(4),
    valAccuracy: (trainingHistory.valAccuracy[index] * 100)?.toFixed(1)
  }));
  
  return (
    <div className="bg-gray-800 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Training
        </h2>
        <InfoTooltip 
          text="Configure hyperparameters and train the CNN model on MNIST data. Training runs entirely in your browser using TensorFlow.js - no data is sent to any server."
          position="left"
        />
      </div>
      
      {/* Data Loading Section */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Data Source</label>
        <div className="flex gap-2">
          <DisabledTooltip
            message={loadDataDisabledReason}
            show={!!loadDataDisabledReason}
            position="top"
          >
            <select
              value={dataSource}
              onChange={(e) => setDataSource(e.target.value)}
              disabled={isTraining || dataLoaded || isLoadingData}
              className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value="sample">Sample Data (~500 images)</option>
              <option value="full">Full MNIST (~60k images)</option>
            </select>
          </DisabledTooltip>
          <DisabledTooltip
            message={loadDataDisabledReason}
            show={!!loadDataDisabledReason}
            position="top"
          >
            <button
              onClick={handleLoadData}
              disabled={isTraining || dataLoaded || isLoadingData}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition-colors cursor-pointer flex items-center gap-2"
            >
              {isLoadingData ? (
                <>
                  <Spinner size="sm" />
                  Loading...
                </>
              ) : dataLoaded ? (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Loaded
                </>
              ) : (
                'Load Data'
              )}
            </button>
          </DisabledTooltip>
        </div>
        
        {(dataLoadProgress > 0 && dataLoadProgress < 1) || isLoadingData ? (
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(dataLoadProgress * 100, 5)}%` }}
            />
          </div>
        ) : null}
        
        {dataLoaded && (
          <p className="text-xs text-green-400 flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            {useFullDataset ? 'Full MNIST dataset (60k images)' : 'Sample dataset (500 images)'} loaded
          </p>
        )}
      </div>
      
      {/* Hyperparameters */}
      <div className="grid grid-cols-3 gap-3">
        <DisabledTooltip
          message={isTraining ? 'Training in progress' : null}
          show={isTraining}
          position="top"
        >
          <div>
            <label className="text-xs text-gray-400 block mb-1">Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
              min={1}
              max={100}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
        </DisabledTooltip>
        <DisabledTooltip
          message={isTraining ? 'Training in progress' : null}
          show={isTraining}
          position="top"
        >
          <div>
            <label className="text-xs text-gray-400 block mb-1">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
              min={1}
              max={512}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
        </DisabledTooltip>
        <DisabledTooltip
          message={isTraining ? 'Training in progress' : null}
          show={isTraining}
          position="top"
        >
          <div>
            <label className="text-xs text-gray-400 block mb-1">Learning Rate</label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
              step={0.0001}
              min={0.0001}
              max={0.1}
              disabled={isTraining}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
        </DisabledTooltip>
      </div>
      
      {/* Training Controls */}
      <div className="flex gap-2">
        {!isTraining ? (
          <DisabledTooltip
            message={trainDisabledReason}
            show={!!trainDisabledReason}
            position="top"
          >
            <button
              onClick={handleStartTraining}
              disabled={!dataLoaded}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
              </svg>
              {model ? 'Continue Training' : 'Start Training'}
            </button>
          </DisabledTooltip>
        ) : (
          <>
            <button
              onClick={onPause}
              className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
            >
              {isPaused ? (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                  </svg>
                  Resume
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M5.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75A.75.75 0 007.25 3h-1.5zM12.75 3a.75.75 0 00-.75.75v12.5c0 .414.336.75.75.75h1.5a.75.75 0 00.75-.75V3.75a.75.75 0 00-.75-.75h-1.5z" />
                  </svg>
                  Pause
                </>
              )}
            </button>
            <button
              onClick={onStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5.75 3A2.75 2.75 0 003 5.75v8.5A2.75 2.75 0 005.75 17h8.5A2.75 2.75 0 0017 14.25v-8.5A2.75 2.75 0 0014.25 3h-8.5z" />
              </svg>
              Stop
            </button>
          </>
        )}
      </div>
      
      {/* Training Progress */}
      {isTraining && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400 flex items-center gap-2">
              <Spinner size="sm" />
              Epoch {trainingProgress.epoch} - Batch {trainingProgress.batch}/{trainingProgress.totalBatches}
            </span>
            <span className="text-white">
              Loss: {trainingProgress.loss.toFixed(4)} | Acc: {(trainingProgress.accuracy * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(trainingProgress.batch / trainingProgress.totalBatches) * 100}%` }}
            />
          </div>
        </div>
      )}
      
      {/* Training Charts */}
      {chartData.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-300">Training Metrics</h3>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="epoch" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#EF4444" name="Train Loss" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="valLoss" stroke="#F97316" name="Val Loss" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="epoch" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#F3F4F6' }}
                />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#10B981" name="Train Acc %" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="valAccuracy" stroke="#06B6D4" name="Val Acc %" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
