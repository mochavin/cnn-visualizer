import * as tf from '@tensorflow/tfjs';

/**
 * Create the CNN model for MNIST digit classification
 */
export function createModel() {
  const model = tf.sequential();
  
  // First convolutional layer
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 8,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    name: 'conv1'
  }));
  
  // First max pooling layer
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
    name: 'pool1'
  }));
  
  // Second convolutional layer
  model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    name: 'conv2'
  }));
  
  // Second max pooling layer
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
    name: 'pool2'
  }));
  
  // Third convolutional layer
  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    padding: 'same',
    activation: 'relu',
    name: 'conv3'
  }));
  
  // Flatten for dense layers
  model.add(tf.layers.flatten({
    name: 'flatten'
  }));
  
  // Dense hidden layer
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    name: 'dense1'
  }));
  
  // Dropout for regularization
  model.add(tf.layers.dropout({
    rate: 0.25,
    name: 'dropout'
  }));
  
  // Output layer
  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
    name: 'output'
  }));
  
  return model;
}

/**
 * Compile the model with optimizer and loss
 */
export function compileModel(model, learningRate = 0.001) {
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

/**
 * Get model architecture summary as array
 */
export function getModelSummary(model) {
  const layers = [];
  
  model.layers.forEach((layer, index) => {
    const config = layer.getConfig();
    const outputShape = layer.outputShape;
    
    layers.push({
      index,
      name: layer.name,
      type: layer.getClassName(),
      outputShape: Array.isArray(outputShape[0]) ? outputShape[0] : outputShape,
      params: layer.countParams(),
      config: {
        filters: config.filters,
        kernelSize: config.kernelSize,
        activation: config.activation,
        units: config.units,
        rate: config.rate,
        poolSize: config.poolSize
      }
    });
  });
  
  return layers;
}

/**
 * Save model to browser downloads
 */
export async function downloadModel(model, name = 'mnist-cnn') {
  await model.save(`downloads://${name}`);
}

/**
 * Load model from uploaded files
 */
export async function loadModelFromFiles(jsonFile, weightsFiles) {
  const model = await tf.loadLayersModel(
    tf.io.browserFiles([jsonFile, ...weightsFiles])
  );
  return model;
}

/**
 * Load pre-trained model from public folder
 */
export async function loadPretrainedModel() {
  const model = await tf.loadLayersModel('/pretrained-model/model.json');
  return model;
}

/**
 * Get the number of trainable parameters
 */
export function getTrainableParams(model) {
  let total = 0;
  model.layers.forEach(layer => {
    total += layer.countParams();
  });
  return total;
}
