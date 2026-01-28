import * as tf from '@tensorflow/tfjs';

// MNIST data constants
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const IMAGE_SIZE = 784; // 28x28
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export interface MNISTData {
  trainImages: tf.Tensor4D;
  trainLabels: tf.Tensor2D;
  testImages: tf.Tensor4D;
  testLabels: tf.Tensor2D;
  numTrain: number;
  numTest: number;
}

/**
 * Load sample MNIST data (bundled, smaller subset for quick testing)
 */
export async function loadSampleMNIST(): Promise<MNISTData> {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}sample-mnist.json`);
    const data = await response.json();
    
    // Flatten 2D images to 1D arrays, then create 4D tensor
    const flattenImage = (img: number[] | number[][]): number[] => {
      if (Array.isArray(img[0])) {
        // It's a 2D array, flatten it
        return (img as number[][]).flat();
      }
      return img as number[]; // Already flat
    };
    
    const trainImagesFlat = data.trainImages.map(flattenImage).flat();
    const testImagesFlat = data.testImages.map(flattenImage).flat();
    
    const numTrain = data.trainImages.length;
    const numTest = data.testImages.length;
    
    const trainImages = tf.tensor4d(
      new Float32Array(trainImagesFlat), 
      [numTrain, 28, 28, 1]
    );
    const testImages = tf.tensor4d(
      new Float32Array(testImagesFlat), 
      [numTest, 28, 28, 1]
    );
    
    const trainLabels = tf.oneHot(
      tf.tensor1d(data.trainLabels, 'int32'), 
      NUM_CLASSES
    ) as tf.Tensor2D;
    const testLabels = tf.oneHot(
      tf.tensor1d(data.testLabels, 'int32'), 
      NUM_CLASSES
    ) as tf.Tensor2D;
    
    return {
      trainImages,
      trainLabels,
      testImages,
      testLabels,
      numTrain,
      numTest
    };
  } catch (error) {
    console.error('Failed to load sample MNIST:', error);
    throw error;
  }
}

/**
 * Load full MNIST dataset from CDN
 */
export async function loadFullMNIST(onProgress?: (p: number) => void): Promise<MNISTData> {
  // Load images
  const img = new Image();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  if (!ctx) throw new Error('Could not get 2D context');

  const imgRequest = new Promise<Float32Array>((resolve, reject) => {
    img.crossOrigin = '';
    img.onload = () => {
      img.width = img.naturalWidth;
      img.height = img.naturalHeight;

      const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

      const chunkSize = 5000;
      canvas.width = img.width;
      canvas.height = chunkSize;

      for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
        const datasetBytesView = new Float32Array(
          datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4, IMAGE_SIZE * chunkSize
        );
        ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        for (let j = 0; j < imageData.data.length / 4; j++) {
          // Normalize to [0, 1]
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
        
        if (onProgress) {
          onProgress((i + 1) / (NUM_DATASET_ELEMENTS / chunkSize) * 0.5);
        }
      }
      resolve(new Float32Array(datasetBytesBuffer));
    };
    img.onerror = reject;
    img.src = MNIST_IMAGES_SPRITE_PATH;
  });

  // Load labels
  const labelsRequest = fetch(MNIST_LABELS_PATH)
    .then(response => response.arrayBuffer())
    .then(buffer => new Uint8Array(buffer));

  const [datasetImages, datasetLabels] = await Promise.all([imgRequest, labelsRequest]);
  
  if (onProgress) onProgress(0.75);

  // Split into train and test
  const trainImagesData = datasetImages.slice(0, NUM_TRAIN_ELEMENTS * IMAGE_SIZE);
  const testImagesData = datasetImages.slice(NUM_TRAIN_ELEMENTS * IMAGE_SIZE);
  
  // Labels are one-hot encoded as bytes (10 bytes per image) in the remote file
  const trainLabelsData = datasetLabels.slice(0, NUM_TRAIN_ELEMENTS * NUM_CLASSES);
  const testLabelsData = datasetLabels.slice(NUM_TRAIN_ELEMENTS * NUM_CLASSES);

  // Create tensors
  const trainImagesTensor = tf.tensor4d(trainImagesData, [NUM_TRAIN_ELEMENTS, 28, 28, 1]);
  const testImagesTensor = tf.tensor4d(testImagesData, [NUM_TEST_ELEMENTS, 28, 28, 1]);
  
  // Labels are already one-hot, just create 2D tensors and cast to float32
  const trainLabelsTensor = tf.tensor2d(new Float32Array(trainLabelsData), [NUM_TRAIN_ELEMENTS, NUM_CLASSES]);
  const testLabelsTensor = tf.tensor2d(new Float32Array(testLabelsData), [NUM_TEST_ELEMENTS, NUM_CLASSES]);
  
  if (onProgress) onProgress(1);

  return {
    trainImages: trainImagesTensor,
    trainLabels: trainLabelsTensor,
    testImages: testImagesTensor,
    testLabels: testLabelsTensor,
    numTrain: NUM_TRAIN_ELEMENTS,
    numTest: NUM_TEST_ELEMENTS
  };
}

/**
 * Get a batch of data for training
 */
export function getBatch(images: tf.Tensor4D, labels: tf.Tensor2D, batchSize: number, batchIndex: number) {
  const numExamples = images.shape[0];
  const startIndex = (batchIndex * batchSize) % numExamples;
  const endIndex = Math.min(startIndex + batchSize, numExamples);
  
  return {
    images: images.slice([startIndex, 0, 0, 0], [endIndex - startIndex, 28, 28, 1]),
    labels: labels.slice([startIndex, 0], [endIndex - startIndex, NUM_CLASSES])
  };
}

/**
 * Shuffle data tensors
 */
export function shuffleData(images: tf.Tensor4D, labels: tf.Tensor2D) {
  const numExamples = images.shape[0];
  const indices = tf.util.createShuffledIndices(numExamples);
  const indicesArray = Array.from(indices);
  
  return tf.tidy(() => {
    const shuffledImages = tf.gather(images, indicesArray);
    const shuffledLabels = tf.gather(labels, indicesArray);
    return { images: shuffledImages, labels: shuffledLabels };
  });
}
