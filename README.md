# CNN Visualizer

An interactive web application for training and visualizing Convolutional Neural Networks (CNNs) directly in your browser. Built with TensorFlow.js, React, and TypeScript.

## Overview

CNN Visualizer allows you to:

- Train a CNN model on the MNIST handwritten digit dataset
- Draw digits and see real-time predictions
- Visualize feature maps, filters, and class activation maps (CAM)
- Understand how neural networks process and classify images

All computation runs entirely in your browser - no server required.

## Features

### Model Training

- Train a CNN from scratch on MNIST data
- Support for both sample dataset (1,000 images) and full dataset (60,000 images)
- Adjustable hyperparameters: learning rate, epochs, batch size
- Real-time training progress with loss and accuracy charts
- Pause/resume training functionality
- Save and load trained models

### Digit Recognition

- Draw digits using mouse or touch input
- Upload image files for recognition
- View prediction confidence for all digit classes (0-9)
- Switch between untrained, pre-trained, and user-trained models

### Visualizations

- **Feature Maps / Activations**: See how each convolutional layer transforms the input image
- **Filter Viewer**: Inspect the learned filters/kernels in each convolutional layer
- **Class Activation Maps (CAM)**: Visualize which regions of the input image contribute most to the prediction
- **Network Diagram**: View the model architecture with layer details

## Tech Stack

- **React 19** - UI framework
- **TypeScript** - Type-safe development
- **TensorFlow.js** - Machine learning in the browser
- **Tailwind CSS 4** - Styling
- **Vite** - Build tool and dev server
- **Recharts** - Training progress charts

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/mochavin/cnn-visualizer.git
cd cnn-visualizer
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Usage

1. **Select a Model**: Choose between an untrained model (random weights) or a pre-trained model with 99% accuracy on MNIST.

2. **Draw a Digit**: Use the drawing canvas to write a digit (0-9). The model will automatically make a prediction.

3. **Train Your Own Model** (optional):
   - Click "Load Sample Data" to load the MNIST dataset
   - Configure training parameters (learning rate, epochs, batch size)
   - Click "Start Training" to begin training
   - Save your trained model to use later

4. **Explore Visualizations**: As you draw or make predictions, observe:
   - Feature maps showing layer activations
   - Convolutional filters learned by the model
   - CAM highlighting important regions for classification

## Project Structure

```
src/
  App.tsx                 # Main application component
  components/
    CAMViewer.tsx         # Class activation map visualization
    DrawingCanvas.tsx     # Digit drawing interface
    FeatureMapViewer.tsx  # Feature map display
    FilterViewer.tsx      # Convolutional filter viewer
    ModelControls.tsx     # Save/load model controls
    NetworkDiagram.tsx    # Model architecture visualization
    TrainingPanel.tsx     # Training configuration and progress
    ui/                   # Reusable UI components
  hooks/
    useModel.ts           # Model state and training logic
  utils/
    mnistLoader.ts        # MNIST data loading utilities
    modelUtils.ts         # Model creation and management
    visualizationUtils.ts # Visualization helper functions
public/
  sample-mnist.json       # Sample MNIST data
  pretrained-model/       # Pre-trained model weights