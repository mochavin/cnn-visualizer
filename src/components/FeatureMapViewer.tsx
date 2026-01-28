import { useEffect, useRef } from 'react';
import { drawToCanvas } from '../utils/visualizationUtils';
import InfoTooltip from './ui/InfoTooltip';
import { LayerOutput } from '../hooks/useModel';

interface FeatureMapViewerProps {
  layerOutputs: Record<string, LayerOutput>;
  title?: string;
}

export default function FeatureMapViewer({ layerOutputs, title = "Feature Maps" }: FeatureMapViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (!layerOutputs || Object.keys(layerOutputs).length === 0) return;
    
    // Clear previous content
    const container = containerRef.current;
    if (!container) return;
    
    container.innerHTML = '';
    
    // Process each layer
    Object.entries(layerOutputs).forEach(([layerName, layerData]) => {
      const { tensor, shape } = layerData;
      
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
        if (tensor.rank === 4) {
          // Convolutional/Pooling layers: [batch, height, width, filters]
          const data = tensor.arraySync() as number[][][][];
          const batchData = data[0]; // First batch item
          const numFilters = batchData[0][0].length;
          const height = batchData.length;
          const width = batchData[0].length;
          
          // Create canvas for each filter
          for (let f = 0; f < Math.min(numFilters, 32); f++) {
            // Extract single filter activation
            const filterData: number[][] = [];
            for (let y = 0; y < height; y++) {
              filterData[y] = [];
              for (let x = 0; x < width; x++) {
                filterData[y][x] = batchData[y][x][f];
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
        } else if (tensor.rank === 2) {
          // Dense/Flatten layers: [batch, units]
          const data = tensor.arraySync() as number[][];
          const batchData = data[0]; // [units]
          const numUnits = batchData.length;
          
          // Force into a single row as requested
          const width = numUnits;
          const height = 1;
          const gridData = [batchData];
          
          // Normalize to [0, 255]
          const flatData = batchData;
          const min = Math.min(...flatData);
          const max = Math.max(...flatData);
          const range = max - min || 1;
          
          const normalizedData = gridData.map(row =>
            row.map(val => ((val - min) / range) * 255)
          );
          
          // Create canvas
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          
          // Discrete look using pixelated rendering
          canvas.style.imageRendering = 'pixelated';
          
          if (layerName === 'output') {
            // Make output units larger and very discrete
            canvas.style.width = '100%';
            canvas.style.height = '32px';
            canvas.style.maxWidth = '300px';
            canvas.className = 'rounded border border-gray-700 bg-black shadow-inner';
            canvas.title = 'Final class probabilities (0-9)';
          } else {
            // For flatten and other dense layers, make it a thin strip
            canvas.style.width = '100%';
            canvas.style.height = '16px';
            canvas.className = 'rounded border border-gray-700 bg-black';
          }
          
          drawToCanvas(canvas, normalizedData, 'viridis');
          grid.appendChild(canvas);
          
          // Add detailed info text
          const infoContainer = document.createElement('div');
          infoContainer.className = 'w-full mt-1 flex justify-between items-center';
          
          const unitInfo = document.createElement('span');
          unitInfo.className = 'text-[10px] text-gray-500 font-mono';
          unitInfo.textContent = `${numUnits} units (1D Vector)`;
          
          const rangeInfo = document.createElement('span');
          rangeInfo.className = 'text-[10px] text-gray-400';
          rangeInfo.textContent = `Range: [${min.toFixed(2)}, ${max.toFixed(2)}]`;
          
          infoContainer.appendChild(unitInfo);
          infoContainer.appendChild(rangeInfo);
          section.appendChild(infoContainer);
          
          if (layerName === 'output') {
            const labelsContainer = document.createElement('div');
            labelsContainer.className = 'w-full flex justify-between px-1 mt-0.5';
            labelsContainer.style.maxWidth = '300px';
            for (let i = 0; i < 10; i++) {
              const lbl = document.createElement('span');
              lbl.className = 'text-[9px] text-gray-600';
              lbl.textContent = i.toString();
              labelsContainer.appendChild(lbl);
            }
            section.appendChild(labelsContainer);
          }
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
            text="Activation outputs from all major layers. For convolutional layers, it shows feature maps. For Flatten and Dense layers, it shows the vector reshaped into a grid. Brighter areas indicate stronger activations."
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
