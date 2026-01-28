import { useState } from 'react';
import InfoTooltip from './ui/InfoTooltip';
import { LayerSummary } from '../utils/modelUtils';

interface NetworkDiagramProps {
  modelSummary: LayerSummary[];
  activeLayer?: string | null;
}

export default function NetworkDiagram({ modelSummary, activeLayer = null }: NetworkDiagramProps) {
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);
  
  if (!modelSummary || modelSummary.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">
            Network Architecture
          </h2>
          <InfoTooltip 
            text="Visual representation of the neural network layers. Each layer transforms the input step by step toward the final prediction."
            position="left"
          />
        </div>
        <p className="text-gray-400 text-sm">Initialize model to see architecture</p>
      </div>
    );
  }
  
  const getLayerColor = (type: string) => {
    switch (type) {
      case 'Conv2D': return 'bg-blue-500 border-blue-400 hover:bg-blue-500';
      case 'MaxPooling2D': return 'bg-purple-500 border-purple-400 hover:bg-purple-500';
      case 'Flatten': return 'bg-yellow-500 border-yellow-400 hover:bg-yellow-500';
      case 'Dense': return 'bg-green-500 border-green-400 hover:bg-green-500';
      case 'Dropout': return 'bg-red-500 border-red-400 hover:bg-red-500';
      default: return 'bg-gray-500 border-gray-400 hover:bg-gray-500';
    }
  };
  
  const formatShape = (shape: number | number[]) => {
    if (!shape) return '';
    if (typeof shape === 'number') return shape.toString();
    return shape.filter(s => s !== null).join(' x ');
  };
  
  const getLayerDetails = (layer: LayerSummary) => {
    const config = layer.config;
    const details = [];
    
    if (config.filters) details.push(`${config.filters} filters`);
    if (config.kernelSize) {
      const ks = Array.isArray(config.kernelSize) ? config.kernelSize.join('x') : config.kernelSize;
      details.push(`${ks} kernel`);
    }
    if (config.poolSize) {
      const ps = Array.isArray(config.poolSize) ? config.poolSize.join('x') : config.poolSize;
      details.push(`${ps} pool`);
    }
    if (config.units) details.push(`${config.units} units`);
    if (config.rate) details.push(`${(config.rate * 100).toFixed(0)}% dropout`);
    if (config.activation) details.push(config.activation);
    
    return details.join(', ');
  };
  
  // Detailed layer descriptions
  const getLayerDescription = (layer: LayerSummary) => {
    const config = layer.config;
    
    switch (layer.type) {
      case 'Conv2D':
        return {
          title: 'Convolutional Layer',
          description: `Applies ${config.filters} learnable filters (${Array.isArray(config.kernelSize) ? config.kernelSize.join('x') : (config.kernelSize || '3x3')}) across the input to detect spatial patterns like edges, textures, and shapes.`,
          details: [
            { label: 'Filters', value: config.filters },
            { label: 'Kernel Size', value: Array.isArray(config.kernelSize) ? config.kernelSize.join(' x ') : config.kernelSize },
            { label: 'Activation', value: config.activation || 'linear' },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: layer.params.toLocaleString() }
          ]
        };
      case 'MaxPooling2D':
        return {
          title: 'Max Pooling Layer',
          description: `Reduces spatial dimensions by taking the maximum value in each ${Array.isArray(config.poolSize) ? config.poolSize.join('x') : (config.poolSize || '2x2')} region. Makes the network more robust to small translations.`,
          details: [
            { label: 'Pool Size', value: Array.isArray(config.poolSize) ? config.poolSize.join(' x ') : config.poolSize },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: '0 (no learnable params)' }
          ]
        };
      case 'Flatten':
        return {
          title: 'Flatten Layer',
          description: 'Converts the 2D feature maps into a 1D vector to feed into fully connected (Dense) layers.',
          details: [
            { label: 'Input Shape', value: '3D tensor (height x width x channels)' },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: '0 (reshaping only)' }
          ]
        };
      case 'Dense':
        return {
          title: 'Fully Connected Layer',
          description: `A layer with ${config.units} neurons, each connected to all inputs. Used for learning complex patterns and making final predictions.`,
          details: [
            { label: 'Units', value: config.units },
            { label: 'Activation', value: config.activation || 'linear' },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: layer.params.toLocaleString() }
          ]
        };
      case 'Dropout':
        return {
          title: 'Dropout Layer',
          description: `Randomly sets ${((config.rate || 0) * 100).toFixed(0)}% of inputs to zero during training. Prevents overfitting by forcing the network to learn redundant representations.`,
          details: [
            { label: 'Dropout Rate', value: `${((config.rate || 0) * 100).toFixed(0)}%` },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Note', value: 'Only active during training' }
          ]
        };
      default:
        return {
          title: layer.type,
          description: 'A neural network layer.',
          details: [
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: layer.params.toLocaleString() }
          ]
        };
    }
  };
  
  const totalParams = modelSummary.reduce((sum, layer) => sum + layer.params, 0);
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white">
          Network Architecture
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 bg-gray-700 px-2 py-1 rounded">
            {totalParams.toLocaleString()} params
          </span>
          <InfoTooltip 
            text="Visual representation of the neural network layers. Hover over each layer to see detailed information about its function and parameters."
            position="left"
          />
        </div>
      </div>
      
      {/* Input */}
      <div className="flex flex-col items-center">
        <div className="px-4 py-2 bg-gray-700 border-2 border-gray-500 rounded-lg text-center mb-2">
          <div className="text-sm font-medium text-white">Input</div>
          <div className="text-xs text-gray-400">28 x 28 x 1</div>
        </div>
        <div className="w-0.5 h-4 bg-gray-600" />
      </div>
      
      {/* Layers */}
      <div className="flex flex-col items-center">
        {modelSummary.map((layer, index) => {
          const layerInfo = getLayerDescription(layer);
          const isHovered = hoveredLayer === layer.name;
          
          return (
            <div key={layer.name} className="flex flex-col items-center w-full relative">
              <div
                className={`w-full max-w-xs px-4 py-2 border-2 rounded-lg cursor-pointer transition-all ${getLayerColor(layer.type)} ${
                  activeLayer === layer.name ? 'ring-2 ring-white ring-offset-2 ring-offset-gray-800' : ''
                } ${isHovered ? 'scale-105 shadow-lg z-10' : ''}`}
                onMouseEnter={() => setHoveredLayer(layer.name)}
                onMouseLeave={() => setHoveredLayer(null)}
              >
                <div className="flex items-center gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">{layer.name}</div>
                    <div className="text-xs text-gray-200 opacity-75">{layer.type}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-200">{formatShape(layer.outputShape)}</div>
                    <div className="text-xs text-gray-300 opacity-75">{layer.params.toLocaleString()}</div>
                  </div>
                </div>
                {getLayerDetails(layer) && (
                  <div className="mt-1 text-xs text-gray-200 opacity-75 truncate">
                    {getLayerDetails(layer)}
                  </div>
                )}
              </div>
              
              {/* Tooltip on hover */}
              {isHovered && (
                <div className="absolute right-full ml-4 top-0 z-50 w-72 bg-gray-900 border border-gray-600 rounded-lg shadow-xl p-3 text-sm">
                  <div className="font-semibold text-white mb-1">{layerInfo.title}</div>
                  <p className="text-gray-300 text-xs mb-2">{layerInfo.description}</p>
                  <div className="space-y-1 border-t border-gray-700 pt-2">
                    {layerInfo.details.map((detail, i) => (
                      <div key={i} className="flex justify-between text-xs">
                        <span className="text-gray-400">{detail.label}:</span>
                        <span className="text-white font-medium">{detail.value}</span>
                      </div>
                    ))}
                  </div>
                  {/* Arrow */}
                  <div className="absolute left-full top-4 w-0 h-0 border-t-8 border-b-8 border-l-8 border-t-transparent border-b-transparent border-l-gray-900" />
                </div>
              )}
              
              {index < modelSummary.length - 1 && (
                <div className="w-0.5 h-3 bg-gray-600" />
              )}
            </div>
          );
        })}
      </div>
      
      {/* Output */}
      <div className="flex flex-col items-center mt-2">
        <div className="w-0.5 h-4 bg-gray-600" />
        <div className="px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 border-2 border-green-400 rounded-lg text-center mt-1">
          <div className="text-sm font-medium text-white">Output</div>
          <div className="text-xs text-green-200">10 classes (digits 0-9)</div>
        </div>
      </div>
      
      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex flex-wrap gap-2 justify-center">
          {['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dropout'].map(type => (
            <div key={type} className="flex items-center gap-1">
              <div className={`w-3 h-3 rounded ${getLayerColor(type).split(' ')[0]}`} />
              <span className="text-xs text-gray-400">{type}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-500 text-center mt-2">Hover over layers for details</p>
      </div>
    </div>
  );
}
