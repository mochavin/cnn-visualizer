import { useEffect, useRef } from 'react';
import { tensorToImageData, drawToCanvas } from '../utils/visualizationUtils';
import InfoTooltip from './ui/InfoTooltip';

export default function FeatureMapViewer({ layerOutputs, title = "Feature Maps" }) {
  const containerRef = useRef(null);
  
  useEffect(() => {
    if (!layerOutputs || Object.keys(layerOutputs).length === 0) return;
    
    // Clear previous content
    const container = containerRef.current;
    if (!container) return;
    
    container.innerHTML = '';
    
    // Process each layer
    Object.entries(layerOutputs).forEach(([layerName, layerData]) => {
      const { tensor, shape, type } = layerData;
      
      // Create layer section
      const section = document.createElement('div');
      section.className = 'mb-4';
      
      // Layer header
      const header = document.createElement('div');
      header.className = 'flex items-center justify-between mb-2';
      header.innerHTML = `
        <span class="text-sm font-medium text-gray-300">${layerName}</span>
        <span class="text-xs text-gray-500">${shape.slice(1).join(' x ')}</span>
      `;
      section.appendChild(header);
      
      // Feature maps grid
      const grid = document.createElement('div');
      grid.className = 'flex flex-wrap gap-1';
      
      try {
        // Get the tensor data
        const data = tensor.arraySync()[0]; // First batch item
        const numFilters = data[0][0].length;
        const height = data.length;
        const width = data[0].length;
        
        // Create canvas for each filter
        for (let f = 0; f < Math.min(numFilters, 32); f++) {
          // Extract single filter activation
          const filterData = [];
          for (let y = 0; y < height; y++) {
            filterData[y] = [];
            for (let x = 0; x < width; x++) {
              filterData[y][x] = data[y][x][f];
            }
          }
          
          // Normalize to [0, 255]
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
          canvas.style.width = `${Math.max(28, width * 2)}px`;
          canvas.style.height = `${Math.max(28, height * 2)}px`;
          canvas.className = 'rounded border border-gray-700 bg-black cursor-pointer hover:border-blue-500 transition-colors';
          canvas.title = `Filter ${f + 1} - Click to enlarge`;
          
          drawToCanvas(canvas, normalizedData, 'viridis');
          grid.appendChild(canvas);
        }
        
        if (numFilters > 32) {
          const moreText = document.createElement('span');
          moreText.className = 'text-xs text-gray-500 self-center px-2';
          moreText.textContent = `+${numFilters - 32} more`;
          grid.appendChild(moreText);
        }
      } catch (e) {
        console.warn(`Failed to visualize ${layerName}:`, e);
        const errorText = document.createElement('span');
        errorText.className = 'text-xs text-red-400';
        errorText.textContent = 'Failed to visualize';
        grid.appendChild(errorText);
      }
      
      section.appendChild(grid);
      container.appendChild(section);
    });
    
  }, [layerOutputs]);
  
  if (!layerOutputs || Object.keys(layerOutputs).length === 0) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">
            {title}
          </h2>
          <InfoTooltip 
            text="Activation outputs from convolutional layers. Shows what patterns each filter detected in your input image. Brighter areas indicate stronger activations."
            position="left"
          />
        </div>
        <p className="text-gray-400 text-sm">Draw a digit and predict to see activations</p>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white">
          {title}
        </h2>
        <InfoTooltip 
          text="Activation outputs from convolutional layers. Each small image shows what a specific filter detected. Brighter areas indicate stronger activations for that pattern."
          position="left"
        />
      </div>
      <div ref={containerRef} className="max-h-96 overflow-y-auto" />
    </div>
  );
}
