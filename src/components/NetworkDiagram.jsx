import { useState } from 'react';
import InfoTooltip from './ui/InfoTooltip';

export default function NetworkDiagram({ modelSummary, activeLayer = null }) {
  const [hoveredLayer, setHoveredLayer] = useState(null);
  
  if (!modelSummary || modelSummary.length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
            </svg>
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
  
  const getLayerColor = (type) => {
    switch (type) {
      case 'Conv2D': return 'bg-blue-600 border-blue-400 hover:bg-blue-500';
      case 'MaxPooling2D': return 'bg-purple-600 border-purple-400 hover:bg-purple-500';
      case 'Flatten': return 'bg-yellow-600 border-yellow-400 hover:bg-yellow-500';
      case 'Dense': return 'bg-green-600 border-green-400 hover:bg-green-500';
      case 'Dropout': return 'bg-red-600 border-red-400 hover:bg-red-500';
      default: return 'bg-gray-600 border-gray-400 hover:bg-gray-500';
    }
  };
  
  const getLayerIcon = (type) => {
    switch (type) {
      case 'Conv2D':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
        );
      case 'MaxPooling2D':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        );
      case 'Flatten':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        );
      case 'Dense':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        );
      case 'Dropout':
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
          </svg>
        );
    }
  };
  
  const formatShape = (shape) => {
    if (!shape) return '';
    return shape.filter(s => s !== null).join(' x ');
  };
  
  const getLayerDetails = (layer) => {
    const config = layer.config;
    const details = [];
    
    if (config.filters) details.push(`${config.filters} filters`);
    if (config.kernelSize) details.push(`${config.kernelSize.join('x')} kernel`);
    if (config.poolSize) details.push(`${config.poolSize.join('x')} pool`);
    if (config.units) details.push(`${config.units} units`);
    if (config.rate) details.push(`${(config.rate * 100).toFixed(0)}% dropout`);
    if (config.activation) details.push(config.activation);
    
    return details.join(', ');
  };
  
  // Detailed layer descriptions
  const getLayerDescription = (layer) => {
    const config = layer.config;
    
    switch (layer.type) {
      case 'Conv2D':
        return {
          title: 'Convolutional Layer',
          description: `Applies ${config.filters} learnable filters (${config.kernelSize?.join('x') || '3x3'}) across the input to detect spatial patterns like edges, textures, and shapes.`,
          details: [
            { label: 'Filters', value: config.filters },
            { label: 'Kernel Size', value: config.kernelSize?.join(' x ') },
            { label: 'Activation', value: config.activation || 'linear' },
            { label: 'Output Shape', value: formatShape(layer.outputShape) },
            { label: 'Parameters', value: layer.params.toLocaleString() }
          ]
        };
      case 'MaxPooling2D':
        return {
          title: 'Max Pooling Layer',
          description: `Reduces spatial dimensions by taking the maximum value in each ${config.poolSize?.join('x') || '2x2'} region. Makes the network more robust to small translations.`,
          details: [
            { label: 'Pool Size', value: config.poolSize?.join(' x ') },
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
          description: `Randomly sets ${(config.rate * 100).toFixed(0)}% of inputs to zero during training. Prevents overfitting by forcing the network to learn redundant representations.`,
          details: [
            { label: 'Dropout Rate', value: `${(config.rate * 100).toFixed(0)}%` },
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
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
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
        <svg className="w-3 h-3 text-gray-600 -mt-1" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 14l-5-5h10l-5 5z" />
        </svg>
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
                  {getLayerIcon(layer.type)}
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
                <div className="absolute left-full ml-4 top-0 z-50 w-72 bg-gray-900 border border-gray-600 rounded-lg shadow-xl p-3 text-sm">
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
                  <div className="absolute right-full top-4 w-0 h-0 border-t-8 border-b-8 border-r-8 border-t-transparent border-b-transparent border-r-gray-900" />
                </div>
              )}
              
              {index < modelSummary.length - 1 && (
                <>
                  <div className="w-0.5 h-3 bg-gray-600" />
                  <svg className="w-3 h-3 text-gray-600 -mt-1" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 14l-5-5h10l-5 5z" />
                  </svg>
                </>
              )}
            </div>
          );
        })}
      </div>
      
      {/* Output */}
      <div className="flex flex-col items-center mt-2">
        <div className="w-0.5 h-4 bg-gray-600" />
        <svg className="w-3 h-3 text-gray-600 -mt-1" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 14l-5-5h10l-5 5z" />
        </svg>
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
