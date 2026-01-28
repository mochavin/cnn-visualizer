import * as tf from '@tensorflow/tfjs';

/**
 * Convert a tensor to canvas ImageData for visualization
 */
export function tensorToImageData(tensor, normalize = true) {
  return tf.tidy(() => {
    let data = tensor;
    
    // Normalize to [0, 255]
    if (normalize) {
      const min = data.min();
      const max = data.max();
      data = data.sub(min).div(max.sub(min).add(1e-8)).mul(255);
    }
    
    return data.arraySync();
  });
}

/**
 * Draw a 2D array to a canvas element
 */
export function drawToCanvas(canvas, data, colormap = 'grayscale') {
  const ctx = canvas.getContext('2d');
  const height = data.length;
  const width = data[0].length;
  
  canvas.width = width;
  canvas.height = height;
  
  const imageData = ctx.createImageData(width, height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const value = Math.min(255, Math.max(0, Math.round(data[y][x])));
      
      if (colormap === 'grayscale') {
        imageData.data[idx] = value;
        imageData.data[idx + 1] = value;
        imageData.data[idx + 2] = value;
      } else if (colormap === 'viridis') {
        const [r, g, b] = viridisColor(value / 255);
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
      } else if (colormap === 'hot') {
        const [r, g, b] = hotColor(value / 255);
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
      }
      imageData.data[idx + 3] = 255; // Alpha
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}

/**
 * Viridis colormap approximation
 */
function viridisColor(t) {
  const r = Math.round(255 * Math.max(0, Math.min(1, 0.267 + 0.005 * t + 2.71 * t * t - 2.36 * t * t * t)));
  const g = Math.round(255 * Math.max(0, Math.min(1, -0.005 + 1.37 * t - 0.35 * t * t)));
  const b = Math.round(255 * Math.max(0, Math.min(1, 0.329 + 1.44 * t - 1.80 * t * t + 0.45 * t * t * t)));
  return [r, g, b];
}

/**
 * Hot colormap
 */
function hotColor(t) {
  const r = Math.round(255 * Math.min(1, t * 3));
  const g = Math.round(255 * Math.max(0, Math.min(1, (t - 0.33) * 3)));
  const b = Math.round(255 * Math.max(0, Math.min(1, (t - 0.67) * 3)));
  return [r, g, b];
}

/**
 * Extract feature maps from a layer
 */
export function extractFeatureMaps(model, layerName, inputTensor) {
  return tf.tidy(() => {
    const layer = model.getLayer(layerName);
    const intermediateModel = tf.model({
      inputs: model.inputs,
      outputs: layer.output
    });
    
    const output = intermediateModel.predict(inputTensor);
    return output;
  });
}

/**
 * Get all convolutional layer outputs
 */
export function getAllConvOutputs(model, inputTensor) {
  const outputs = {};
  
  model.layers.forEach((layer, index) => {
    if (layer.getClassName() === 'Conv2D') {
      const intermediateModel = tf.model({
        inputs: model.inputs,
        outputs: layer.output
      });
      
      const output = intermediateModel.predict(inputTensor);
      outputs[`${layer.name} (Layer ${index})`] = output;
    }
  });
  
  return outputs;
}

/**
 * Extract filter weights from a conv layer
 */
export function extractFilters(model, layerName) {
  const layer = model.getLayer(layerName);
  const weights = layer.getWeights()[0]; // [height, width, inChannels, outChannels]
  return weights;
}

/**
 * Create a Grad-CAM visualization
 */
export async function computeGradCAM(model, inputTensor, classIndex) {
  // Find the last conv layer
  let lastConvLayerIndex = -1;
  for (let i = model.layers.length - 1; i >= 0; i--) {
    if (model.layers[i].getClassName() === 'Conv2D') {
      lastConvLayerIndex = i;
      break;
    }
  }
  
  if (lastConvLayerIndex === -1) {
    console.warn('No Conv2D layer found');
    return null;
  }
  
  const lastConvLayer = model.layers[lastConvLayerIndex];
  
  // Create model that outputs both the conv layer output and the final prediction
  const gradModel = tf.model({
    inputs: model.inputs,
    outputs: [lastConvLayer.output, model.output]
  });
  
  // Compute gradients
  const [convOutput, grads] = tf.tidy(() => {
    const tape = tf.grad((x) => {
      const [convOut, predictions] = gradModel.apply(x, { training: false });
      return predictions.gather([classIndex], 1).squeeze();
    });
    
    const gradients = tape(inputTensor);
    const [convOutputVal] = gradModel.predict(inputTensor);
    
    return [convOutputVal, gradients];
  });
  
  // Global average pooling of gradients
  const weights = tf.tidy(() => {
    return grads.mean([1, 2]); // Average over height and width
  });
  
  // Weighted combination of feature maps
  const cam = tf.tidy(() => {
    const convOutputArray = convOutput.squeeze();
    const weightsArray = weights.squeeze();
    
    // Multiply each feature map by its weight and sum
    let weightedSum = tf.zeros([convOutputArray.shape[0], convOutputArray.shape[1]]);
    const numFilters = convOutputArray.shape[2];
    
    for (let i = 0; i < numFilters; i++) {
      const featureMap = convOutputArray.slice([0, 0, i], [-1, -1, 1]).squeeze();
      const weight = weightsArray.slice([i], [1]).squeeze();
      weightedSum = weightedSum.add(featureMap.mul(weight));
    }
    
    // ReLU and normalize
    const camRelu = tf.relu(weightedSum);
    const camNorm = camRelu.div(camRelu.max().add(1e-8));
    
    // Resize to input size
    const camResized = tf.image.resizeBilinear(
      camNorm.expandDims(0).expandDims(-1),
      [28, 28]
    );
    
    return camResized.squeeze();
  });
  
  // Cleanup intermediate tensors
  convOutput.dispose();
  grads.dispose();
  weights.dispose();
  
  return cam;
}

/**
 * Create overlay of CAM on original image
 */
export function createCAMOverlay(originalImage, camData, alpha = 0.6) {
  const height = originalImage.length;
  const width = originalImage[0].length;
  const overlay = [];
  
  for (let y = 0; y < height; y++) {
    overlay[y] = [];
    for (let x = 0; x < width; x++) {
      const originalValue = originalImage[y][x] * 255;
      const camValue = camData[y][x] * 255;
      const [r, g, b] = hotColor(camValue / 255);
      
      // Blend original (grayscale) with CAM (colorized)
      overlay[y][x] = {
        r: Math.round(originalValue * (1 - alpha) + r * alpha),
        g: Math.round(originalValue * (1 - alpha) + g * alpha),
        b: Math.round(originalValue * (1 - alpha) + b * alpha)
      };
    }
  }
  
  return overlay;
}

/**
 * Draw CAM overlay to canvas
 */
export function drawCAMToCanvas(canvas, overlay) {
  const ctx = canvas.getContext('2d');
  const height = overlay.length;
  const width = overlay[0].length;
  
  canvas.width = width;
  canvas.height = height;
  
  const imageData = ctx.createImageData(width, height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      imageData.data[idx] = overlay[y][x].r;
      imageData.data[idx + 1] = overlay[y][x].g;
      imageData.data[idx + 2] = overlay[y][x].b;
      imageData.data[idx + 3] = 255;
    }
  }
  
  ctx.putImageData(imageData, 0, 0);
}
