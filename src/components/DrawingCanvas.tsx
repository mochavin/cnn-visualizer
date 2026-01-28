import { useRef, useEffect, useState, useCallback, MouseEvent, TouchEvent, ChangeEvent } from 'react';
import * as tf from '@tensorflow/tfjs';
import Spinner from './ui/Spinner';
import DisabledTooltip from './ui/DisabledTooltip';

interface DrawingCanvasProps {
  onImageReady: (tensor: tf.Tensor) => void;
  disabled?: boolean;
  isPredicting?: boolean;
}

export default function DrawingCanvas({ onImageReady, disabled = false, isPredicting = false }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  
  // Determine disabled reasons
  const getDisabledReason = () => {
    if (disabled) return 'Training in progress';
    return null;
  };
  
  const getPredictDisabledReason = () => {
    if (disabled) return 'Training in progress';
    if (!hasDrawn) return 'Draw a digit first';
    if (isPredicting) return 'Prediction in progress';
    return null;
  };
  
  const disabledReason = getDisabledReason();
  const predictDisabledReason = getPredictDisabledReason();
  
  const setupContext = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set black background once on mount
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Initial context setup
    setupContext();
  }, [setupContext]);
  
  const getCoordinates = useCallback((e: MouseEvent | TouchEvent | any) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    
    let clientX, clientY;
    if (e.touches) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }
    
    // Scale coordinates based on canvas display size vs actual size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }, []);
  
  const startDrawing = useCallback((e: MouseEvent | TouchEvent) => {
    if (disabled) return;
    e.preventDefault();
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { x, y } = getCoordinates(e);
    
    // Re-setup context to ensure properties persist
    setupContext();
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  }, [disabled, getCoordinates, setupContext]);
  
  const draw = useCallback((e: MouseEvent | TouchEvent) => {
    if (!isDrawing || disabled) return;
    e.preventDefault();
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { x, y } = getCoordinates(e);
    
    ctx.lineTo(x, y);
    ctx.stroke();
    setHasDrawn(true);
  }, [isDrawing, disabled, getCoordinates]);
  
  const stopDrawing = useCallback((e?: MouseEvent | TouchEvent) => {
    if (e) {
      if (typeof e.preventDefault === 'function') e.preventDefault();
    }
    setIsDrawing(false);
  }, []);
  
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Re-setup context after clear
    setupContext();
    
    setHasDrawn(false);
  }, [setupContext]);
  
  const getImageTensor = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) throw new Error('Canvas not found');
    
    // Create a temporary canvas for downscaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) throw new Error('Could not get 2D context');
    
    // Draw scaled down image
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // Convert to grayscale tensor normalized to [0, 1]
    const values = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      // Use red channel (all channels should be same for grayscale)
      values[i] = data[i * 4] / 255;
    }
    
    // Reshape to [1, 28, 28, 1]
    const tensor = tf.tensor4d(values, [1, 28, 28, 1]);
    
    return tensor;
  }, []);
  
  const handlePredict = useCallback(() => {
    if (!hasDrawn || isPredicting) return;
    const tensor = getImageTensor();
    if (onImageReady) {
      onImageReady(tensor);
    }
  }, [hasDrawn, isPredicting, getImageTensor, onImageReady]);
  
  // Handle file upload
  const handleImageUpload = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        
        // Clear canvas
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw image centered and scaled
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const x = (canvas.width - img.width * scale) / 2;
        const y = (canvas.height - img.height * scale) / 2;
        
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
        setHasDrawn(true);
        
        // Trigger prediction
        const tensor = getImageTensor();
        if (onImageReady) {
          onImageReady(tensor);
        }
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  }, [getImageTensor, onImageReady]);
  
  return (
    <div className="flex flex-col gap-3">
      {/* Canvas container with fixed dimensions for proper centering */}
      <DisabledTooltip 
        message={disabledReason || ''}
        show={!!disabledReason}
        position="top"
      >
        <div 
          className="relative mx-auto"
          style={{ width: '280px', height: '280px' }}
        >
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            className={`border-2 border-gray-600 rounded-lg bg-black touch-none w-full h-full ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-crosshair'}`}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
          {!hasDrawn && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <span className="text-gray-500 text-sm text-center px-4">Draw a digit here</span>
            </div>
          )}
        </div>
      </DisabledTooltip>
      
      <div className="flex gap-2">
        <DisabledTooltip 
          message={predictDisabledReason || ''}
          show={!!predictDisabledReason}
          position="top"
        >
          <button
            onClick={handlePredict}
            disabled={!hasDrawn || disabled || isPredicting}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors cursor-pointer flex items-center justify-center gap-2"
          >
            {isPredicting ? (
              <>
                <Spinner size="sm" />
                Predicting...
              </>
            ) : (
              'Predict'
            )}
          </button>
        </DisabledTooltip>
        <DisabledTooltip 
          message={disabledReason || ''}
          show={!!disabledReason}
          position="top"
        >
          <button
            onClick={clearCanvas}
            disabled={disabled}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors cursor-pointer"
          >
            Clear
          </button>
        </DisabledTooltip>
      </div>
      
      <DisabledTooltip 
        message={disabledReason || ''}
        show={!!disabledReason}
        position="top"
      >
        <div className="flex items-center gap-2">
          <label className="flex-1">
            <span className="sr-only">Upload image</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              disabled={disabled}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-gray-700 file:text-white hover:file:bg-gray-600 file:cursor-pointer disabled:opacity-50"
            />
          </label>
        </div>
      </DisabledTooltip>
    </div>
  );
}
