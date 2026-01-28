import { useEffect, useRef, useState } from 'react';
import { drawToCanvas } from '../utils/visualizationUtils';
import InfoTooltip from './ui/InfoTooltip';

export default function FilterViewer({ model, selectedLayer = null }) {
  const containerRef = useRef(null);
  const [availableLayers, setAvailableLayers] = useState([]);
  const [currentLayer, setCurrentLayer] = useState(selectedLayer);
  
  // Get available conv layers
  useEffect(() => {
    if (!model) {
      setAvailableLayers([]);
      return;
    }
    
    const convLayers = model.layers
      .filter(layer => layer.getClassName() === 'Conv2D')
      .map(layer => layer.name);
    
    setAvailableLayers(convLayers);
    
    if (convLayers.length > 0 && !currentLayer) {
      setCurrentLayer(convLayers[0]);
    }
  }, [model, currentLayer]);
  
  // Visualize filters
  useEffect(() => {
    if (!model || !currentLayer) return;
    
    const container = containerRef.current;
    if (!container) return;
    
    container.innerHTML = '';
    
    try {
      const layer = model.getLayer(currentLayer);
      const weights = layer.getWeights()[0]; // [height, width, inChannels, outChannels]
      const weightsData = weights.arraySync();
      
      const [height, width, inChannels, outChannels] = weights.shape;
      
      // Create filter visualizations
      const grid = document.createElement('div');
      grid.className = 'flex flex-wrap gap-2';
      
      for (let f = 0; f < outChannels; f++) {
        const filterDiv = document.createElement('div');
        filterDiv.className = 'flex flex-col items-center';
        
        // For multi-channel inputs, average across input channels
        const filterData = [];
        for (let y = 0; y < height; y++) {
          filterData[y] = [];
          for (let x = 0; x < width; x++) {
            let sum = 0;
            for (let c = 0; c < inChannels; c++) {
              sum += weightsData[y][x][c][f];
            }
            filterData[y][x] = sum / inChannels;
          }
        }
        
        // Normalize
        const flatData = filterData.flat();
        const min = Math.min(...flatData);
        const max = Math.max(...flatData);
        const range = max - min || 1;
        
        const normalizedData = filterData.map(row =>
          row.map(val => ((val - min) / range) * 255)
        );
        
        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = '40px';
        canvas.style.height = '40px';
        canvas.style.imageRendering = 'pixelated';
        canvas.className = 'rounded border border-gray-600 cursor-pointer hover:border-blue-500 transition-colors';
        canvas.title = `Filter ${f + 1} - These weights are applied across the input`;
        
        drawToCanvas(canvas, normalizedData, 'viridis');
        
        // Filter label
        const label = document.createElement('span');
        label.className = 'text-xs text-gray-500 mt-1';
        label.textContent = `F${f + 1}`;
        
        filterDiv.appendChild(canvas);
        filterDiv.appendChild(label);
        grid.appendChild(filterDiv);
      }
      
      container.appendChild(grid);
      
      // Add stats
      const stats = document.createElement('div');
      stats.className = 'mt-3 text-xs text-gray-400 border-t border-gray-700 pt-3';
      stats.innerHTML = `
        <div class="flex gap-4">
          <span>Filters: ${outChannels}</span>
          <span>Size: ${height} x ${width}</span>
          <span>Input Channels: ${inChannels}</span>
        </div>
      `;
      container.appendChild(stats);
      
    } catch (e) {
      console.warn(`Failed to visualize filters for ${currentLayer}:`, e);
      const errorText = document.createElement('span');
      errorText.className = 'text-sm text-red-400';
      errorText.textContent = 'Failed to load filter weights';
      container.appendChild(errorText);
    }
    
  }, [model, currentLayer]);
  
  if (!model) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
            Filter Kernels
          </h2>
          <InfoTooltip 
            text="The learned weights of convolutional filters. These small matrices slide across the input to detect specific patterns like edges, corners, or textures."
            position="left"
          />
        </div>
        <p className="text-gray-400 text-sm">Initialize model to see filter weights</p>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
          </svg>
          Filter Kernels
        </h2>
        
        <div className="flex items-center gap-2">
          <select
            value={currentLayer || ''}
            onChange={(e) => setCurrentLayer(e.target.value)}
            className="px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            {availableLayers.map(layer => (
              <option key={layer} value={layer}>{layer}</option>
            ))}
          </select>
          <InfoTooltip 
            text="The learned weights of convolutional filters. Each filter detects a specific pattern. After training, these weights are optimized to recognize digit features."
            position="left"
          />
        </div>
      </div>
      
      <div ref={containerRef} className="max-h-64 overflow-y-auto" />
    </div>
  );
}
