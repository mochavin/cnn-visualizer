import { useRef, useState, ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs';
import Spinner from './ui/Spinner';
import InfoTooltip from './ui/InfoTooltip';
import { ModelSource } from '../hooks/useModel';

interface ModelControlsProps {
  model: tf.LayersModel | null;
  onSave: (name: string) => Promise<void>;
  onLoad: (jsonFile: File, weightsFiles: File[]) => Promise<tf.LayersModel>;
  modelSource: ModelSource;
  modelName: string | null;
}

export default function ModelControls({ model, onSave, onLoad }: ModelControlsProps) {
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const weightsInputRef = useRef<HTMLInputElement>(null);
  const [jsonFile, setJsonFile] = useState<File | null>(null);
  const [weightsFiles, setWeightsFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  const handleDownload = async () => {
    if (!model) return;
    
    setIsDownloading(true);
    setError(null);
    setSuccess(null);
    
    try {
      await onSave('mnist-cnn');
      setSuccess('Model downloaded successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      console.error('Failed to save model:', e);
      setError('Failed to download model');
    } finally {
      setIsDownloading(false);
    }
  };
  
  const handleJsonChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setJsonFile(file);
      setError(null);
    }
  };
  
  const handleWeightsChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setWeightsFiles(files);
      setError(null);
    }
  };
  
  const handleLoad = async () => {
    if (!jsonFile) {
      setError('Please select model.json file');
      return;
    }
    
    if (weightsFiles.length === 0) {
      setError('Please select weight files (.bin)');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      await onLoad(jsonFile, weightsFiles);
      
      // Reset file inputs
      setJsonFile(null);
      setWeightsFiles([]);
      if (jsonInputRef.current) jsonInputRef.current.value = '';
      if (weightsInputRef.current) weightsInputRef.current.value = '';
      setSuccess('Model loaded successfully!');
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      console.error('Failed to load model:', e);
      setError('Failed to load model. Make sure you selected the correct files.');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          Model I/O
        </h2>
        <InfoTooltip 
          text="Download your trained model to save it locally, or upload a previously saved model to continue using it. Models are saved in TensorFlow.js format (JSON + binary weights)."
          position="left"
        />
      </div>
      
      {/* Success/Error messages */}
      {success && (
        <div className="mb-3 px-3 py-2 bg-green-900/50 border border-green-700 rounded-lg text-sm text-green-400 flex items-center gap-2">
          {success}
        </div>
      )}
      
      {error && (
        <div className="mb-3 px-3 py-2 bg-red-900/50 border border-red-700 rounded-lg text-sm text-red-400 flex items-center gap-2">
          {error}
        </div>
      )}
      
      <div className="space-y-4">
        {/* Download Section */}
        <div>
          <h3 className="text-sm font-medium text-gray-300 mb-2">Download Model</h3>
          <button
            onClick={handleDownload}
            disabled={!model || isDownloading}
            className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
          >
            {isDownloading ? (
              <>
                <Spinner size="sm" />
                Downloading...
              </>
            ) : (
              'Download Model'
            )}
          </button>
          {!model && (
            <p className="text-xs text-gray-500 mt-1">Initialize and train a model first</p>
          )}
        </div>
        
        {/* Upload Section */}
        <div className="border-t border-gray-700 pt-4">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Load Model</h3>
          
          <div className="space-y-2">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Model JSON</label>
              <input
                ref={jsonInputRef}
                type="file"
                accept=".json"
                onChange={handleJsonChange}
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-gray-700 file:text-white hover:file:bg-gray-600 file:cursor-pointer cursor-pointer"
              />
            </div>
            
            <div>
              <label className="text-xs text-gray-400 block mb-1">Weight Files (.bin)</label>
              <input
                ref={weightsInputRef}
                type="file"
                accept=".bin"
                multiple
                onChange={handleWeightsChange}
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-gray-700 file:text-white hover:file:bg-gray-600 file:cursor-pointer cursor-pointer"
              />
            </div>
            
            {(jsonFile || weightsFiles.length > 0) && (
              <div className="text-xs text-gray-400 bg-gray-700 rounded p-2">
                <div>{jsonFile && `JSON: ${jsonFile.name}`}</div>
                <div>{weightsFiles.length > 0 && `Weights: ${weightsFiles.map(f => f.name).join(', ')}`}</div>
              </div>
            )}
            
            <button
              onClick={handleLoad}
              disabled={!jsonFile || weightsFiles.length === 0 || isLoading}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Spinner size="sm" />
                  Loading...
                </>
              ) : (
                'Load Model'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
