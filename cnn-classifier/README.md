# CNN Classifier

**Soft Computing Lab — Experiment 10: Implement a CNN classifier.**

`main.py` builds and trains a convolutional neural network that classifies
handwritten digits from the **MNIST** dataset, using TensorFlow/Keras.

## Network

```
Input 28×28×1
 → Conv2D(32, 3×3, relu) → MaxPool 2×2
 → Conv2D(64, 3×3, relu) → MaxPool 2×2
 → Flatten → Dense(64, relu) → Dropout(0.3)
 → Dense(10, softmax)
```

Convolution layers learn local features (edges, strokes), pooling adds
translation tolerance and shrinks the feature maps, and the dense head classifies
the 10 digits. It trains with the Adam optimiser and sparse categorical
cross-entropy, and typically reaches ~99% test accuracy in a few epochs.

## Setup and run

```bash
pip install -r requirements.txt
python3 main.py
```

MNIST downloads automatically on first run. The script prints the model summary,
trains for 5 epochs, and reports test accuracy.

## Notes

This is the only lab program with a heavy dependency (TensorFlow, in
`requirements.txt`). Training is much faster on a GPU but runs on CPU too.
