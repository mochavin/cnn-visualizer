// This script generates sample MNIST data for the CNN Visualizer
// Run with: node scripts/generateSampleData.js

const https = require('https');
const fs = require('fs');
const path = require('path');

const MNIST_IMAGES_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const SAMPLE_SIZE = 500; // Number of samples to include
const IMAGE_SIZE = 784; // 28x28

async function fetchBuffer(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function generateSampleData() {
  console.log('This script requires canvas package. For now, using pre-generated data.');
  console.log('The sample-mnist.json file should be created manually or downloaded.');
  
  // Create a simple placeholder with synthetic data for testing
  const trainImages = [];
  const trainLabels = [];
  const testImages = [];
  const testLabels = [];
  
  // Generate simple synthetic digit-like patterns for testing
  for (let i = 0; i < 400; i++) {
    const label = i % 10;
    const image = generateDigitPattern(label);
    trainImages.push(image);
    trainLabels.push(label);
  }
  
  for (let i = 0; i < 100; i++) {
    const label = i % 10;
    const image = generateDigitPattern(label);
    testImages.push(image);
    testLabels.push(label);
  }
  
  const data = {
    trainImages,
    trainLabels,
    testImages,
    testLabels
  };
  
  const outputPath = path.join(__dirname, '..', 'public', 'sample-mnist.json');
  fs.writeFileSync(outputPath, JSON.stringify(data));
  console.log(`Generated ${outputPath}`);
  console.log(`Train samples: ${trainImages.length}, Test samples: ${testImages.length}`);
}

function generateDigitPattern(digit) {
  // Create a 28x28 image with a simple pattern for each digit
  const image = [];
  for (let y = 0; y < 28; y++) {
    const row = [];
    for (let x = 0; x < 28; x++) {
      row.push(0);
    }
    image.push(row);
  }
  
  // Draw simple patterns based on digit
  const patterns = {
    0: () => drawCircle(image, 14, 14, 8),
    1: () => drawLine(image, 14, 4, 14, 24),
    2: () => { drawLine(image, 6, 8, 22, 8); drawLine(image, 22, 8, 6, 20); drawLine(image, 6, 20, 22, 20); },
    3: () => { drawLine(image, 6, 6, 20, 6); drawLine(image, 6, 14, 20, 14); drawLine(image, 6, 22, 20, 22); drawLine(image, 20, 6, 20, 22); },
    4: () => { drawLine(image, 6, 6, 6, 14); drawLine(image, 6, 14, 20, 14); drawLine(image, 20, 6, 20, 22); },
    5: () => { drawLine(image, 22, 6, 6, 6); drawLine(image, 6, 6, 6, 14); drawLine(image, 6, 14, 20, 14); drawLine(image, 20, 14, 20, 22); drawLine(image, 20, 22, 6, 22); },
    6: () => { drawCircle(image, 14, 16, 6); drawLine(image, 8, 6, 8, 16); },
    7: () => { drawLine(image, 6, 6, 22, 6); drawLine(image, 22, 6, 14, 22); },
    8: () => { drawCircle(image, 14, 10, 5); drawCircle(image, 14, 18, 5); },
    9: () => { drawCircle(image, 14, 10, 6); drawLine(image, 20, 10, 20, 22); }
  };
  
  if (patterns[digit]) {
    patterns[digit]();
  }
  
  // Add some noise
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (Math.random() < 0.02) {
        image[y][x] = Math.min(1, image[y][x] + Math.random() * 0.3);
      }
    }
  }
  
  return image;
}

function drawCircle(image, cx, cy, r) {
  for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {
    const x = Math.round(cx + Math.cos(angle) * r);
    const y = Math.round(cy + Math.sin(angle) * r);
    if (x >= 0 && x < 28 && y >= 0 && y < 28) {
      image[y][x] = 1;
      // Thicken the line
      if (x > 0) image[y][x-1] = Math.max(image[y][x-1], 0.5);
      if (x < 27) image[y][x+1] = Math.max(image[y][x+1], 0.5);
    }
  }
}

function drawLine(image, x1, y1, x2, y2) {
  const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
  for (let i = 0; i <= steps; i++) {
    const t = steps === 0 ? 0 : i / steps;
    const x = Math.round(x1 + (x2 - x1) * t);
    const y = Math.round(y1 + (y2 - y1) * t);
    if (x >= 0 && x < 28 && y >= 0 && y < 28) {
      image[y][x] = 1;
      // Thicken the line
      if (y > 0) image[y-1][x] = Math.max(image[y-1][x], 0.7);
      if (y < 27) image[y+1][x] = Math.max(image[y+1][x], 0.7);
    }
  }
}

generateSampleData();
