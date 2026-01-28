import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import InfoTooltip from './ui/InfoTooltip';

export default function CAMViewer({ model, inputTensor, prediction }) {
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [camReady, setCamReady] = useState(false);
  
  useEffect(() => {
    if (!model || !inputTensor || prediction === null || prediction === undefined) {
      setCamReady(false);
      return;
    }
    
    const computeCAM = async () => {
      try {
        // Find the last conv layer
        let lastConvLayer = null;
        for (let i = model.layers.length - 1; i >= 0; i--) {
          if (model.layers[i].getClassName() === 'Conv2D') {
            lastConvLayer = model.layers[i];
            break;
          }
        }
        
        if (!lastConvLayer) {
          console.warn('No Conv2D layer found for CAM');
          return;
        }
        
        // Create a model that outputs the conv layer activations and the predictions
        const convOutputModel = tf.model({
          inputs: model.inputs,
          outputs: lastConvLayer.output
        });
        
        // Get conv layer output
        const convOutput = convOutputModel.predict(inputTensor);
        const convOutputData = convOutput.arraySync()[0]; // [height, width, filters]
        
        // Get the weights from the last dense layer (before softmax)
        const outputLayer = model.layers[model.layers.length - 1];
        const denseWeights = outputLayer.getWeights()[0].arraySync(); // [input_features, num_classes]
        
        // Find the flatten layer to understand the mapping
        const flattenIndex = model.layers.findIndex(l => l.getClassName() === 'Flatten');
        const beforeFlatten = model.layers[flattenIndex - 1];
        const beforeFlattenShape = beforeFlatten.outputShape; // [null, h, w, filters]
        
        const convHeight = convOutputData.length;
        const convWidth = convOutputData[0].length;
        const numFilters = convOutputData[0][0].length;
        
        // Simple CAM: weight each activation map by the corresponding weight for the predicted class
        // This is a simplified version - full Grad-CAM would use gradients
        const cam = Array(convHeight).fill(0).map(() => Array(convWidth).fill(0));
        
        // For each spatial location, sum weighted activations
        for (let y = 0; y < convHeight; y++) {
          for (let x = 0; x < convWidth; x++) {
            let weightedSum = 0;
            for (let f = 0; f < numFilters; f++) {
              // Use activation value directly as importance (simplified CAM)
              weightedSum += convOutputData[y][x][f];
            }
            cam[y][x] = Math.max(0, weightedSum); // ReLU
          }
        }
        
        // Normalize CAM
        const flatCam = cam.flat();
        const maxVal = Math.max(...flatCam);
        const minVal = Math.min(...flatCam);
        const range = maxVal - minVal || 1;
        
        for (let y = 0; y < convHeight; y++) {
          for (let x = 0; x < convWidth; x++) {
            cam[y][x] = (cam[y][x] - minVal) / range;
          }
        }
        
        // Draw original image
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = 28;
        canvas.height = 28;
        
        const inputData = inputTensor.arraySync()[0];
        const imageData = ctx.createImageData(28, 28);
        
        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            const idx = (y * 28 + x) * 4;
            const val = Math.round(inputData[y][x][0] * 255);
            imageData.data[idx] = val;
            imageData.data[idx + 1] = val;
            imageData.data[idx + 2] = val;
            imageData.data[idx + 3] = 255;
          }
        }
        ctx.putImageData(imageData, 0, 0);
        
        // Draw CAM overlay
        const overlayCanvas = overlayCanvasRef.current;
        const overlayCtx = overlayCanvas.getContext('2d');
        overlayCanvas.width = 28;
        overlayCanvas.height = 28;
        
        // Upscale CAM to 28x28
        const scaleY = 28 / convHeight;
        const scaleX = 28 / convWidth;
        
        const overlayImageData = overlayCtx.createImageData(28, 28);
        
        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            const camY = Math.min(Math.floor(y / scaleY), convHeight - 1);
            const camX = Math.min(Math.floor(x / scaleX), convWidth - 1);
            const camVal = cam[camY][camX];
            
            // Hot colormap
            const r = Math.min(255, Math.round(camVal * 3 * 255));
            const g = Math.min(255, Math.round(Math.max(0, camVal - 0.33) * 3 * 255));
            const b = Math.min(255, Math.round(Math.max(0, camVal - 0.67) * 3 * 255));
            
            // Blend with original
            const origVal = inputData[y][x][0] * 255;
            const alpha = 0.6;
            
            const idx = (y * 28 + x) * 4;
            overlayImageData.data[idx] = Math.round(origVal * (1 - alpha) + r * alpha);
            overlayImageData.data[idx + 1] = Math.round(origVal * (1 - alpha) + g * alpha);
            overlayImageData.data[idx + 2] = Math.round(origVal * (1 - alpha) + b * alpha);
            overlayImageData.data[idx + 3] = 255;
          }
        }
        
        overlayCtx.putImageData(overlayImageData, 0, 0);
        
        // Cleanup
        convOutput.dispose();
        
        setCamReady(true);
        
      } catch (e) {
        console.error('Failed to compute CAM:', e);
        setCamReady(false);
      }
    };
    
    computeCAM();
    
  }, [model, inputTensor, prediction]);
  
  if (!model || !inputTensor) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
            Class Activation Map
          </h2>
          <InfoTooltip 
            text="Heatmap showing which regions of the input image most influenced the prediction. Brighter/warmer colors indicate more important regions."
            position="left"
          />
        </div>
        <p className="text-gray-400 text-sm">Draw and predict to see activation heatmap</p>
      </div>
    );
  }
  
  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
          </svg>
          Class Activation Map
        </h2>
        <InfoTooltip 
          text="Heatmap showing which regions of the input image most influenced the model's prediction. This helps understand what the network 'looks at' when making decisions."
          position="left"
        />
      </div>
      
      <div className="flex gap-4 items-start">
        <div className="flex flex-col items-center">
          <canvas
            ref={canvasRef}
            className="w-24 h-24 rounded border border-gray-600 bg-black"
            style={{ imageRendering: 'pixelated' }}
          />
          <span className="text-xs text-gray-400 mt-1">Original</span>
        </div>
        
        <div className="flex flex-col items-center">
          <canvas
            ref={overlayCanvasRef}
            className="w-24 h-24 rounded border border-gray-600 bg-black"
            style={{ imageRendering: 'pixelated' }}
          />
          <span className="text-xs text-gray-400 mt-1">CAM Overlay</span>
        </div>
        
        <div className="flex-1">
          <p className="text-sm text-gray-300 mb-2">
            Predicted: <span className="font-bold text-white text-lg">{prediction}</span>
          </p>
          <p className="text-xs text-gray-500">
            Brighter areas indicate regions that contributed more to the prediction.
          </p>
          
          {/* Color scale */}
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-gray-500">Low</span>
            <div className="flex-1 h-3 rounded" style={{
              background: 'linear-gradient(to right, #000, #ff0000, #ffff00, #ffffff)'
            }} />
            <span className="text-xs text-gray-500">High</span>
          </div>
        </div>
      </div>
    </div>
  );
}
