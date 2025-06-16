# HDC Benchmark: Raspberry Pi vs. Google Colab GPU

This project evaluates the performance of **Hyperdimensional Computing (HDC)** for MNIST digit classification on two contrasting platforms:
- Google Colab GPU (NVIDIA Tesla T4)
- Raspberry Pi 4 Model B

We compare the two in terms of:
- Inference Speed
- Memory Usage
- Power Consumption

> Developed as a final project for *ECE 618 – Real-Time Embedded Systems* at George Mason University.

---

## What is Hyperdimensional Computing?

HDC mimics the human brain's use of **high-dimensional binary or bipolar vectors** to represent and process data efficiently. It enables robust, fast, and low-power machine learning suitable for **edge computing**.

---

## Experimental Setup

- **Dataset**: MNIST Handwritten Digit Dataset
- **HDC Model**: 
  - Encoding: bipolar hypervectors
  - Training: bundling (majority voting)
  - Inference: cosine similarity
- **Platforms**:
  - Raspberry Pi 4 Model B (Quad-core Cortex-A72)
  - Google Colab GPU (Tesla T4)

---

## File Structure

```bash
HDCBenchmark-RPivsGPU/
├── HDC_GPU.py # GPU-based implementation (Google Colab)
├── HDC_Rpi.py # Edge implementation (Raspberry Pi)
├── Final Report.pdf # Detailed project documentation
├── Presentation.pptx # Final presentation slides

```


---

## How to Run

### GPU (Google Colab or desktop CUDA system)
  ```bash
  python3 HDC_GPU.py
  ```
### Edge Device (Raspberry Pi)
```bash
python3 HDC_Rpi.py
```

---

## Dependencies

- `numpy`
- `scikit-learn`
- `matplotlib`
- `time`, `os`, `random`
- [Optional] `torch` (if GPU version uses PyTorch for acceleration)

### Install via pip:
```bash
pip install numpy scikit-learn matplotlib

```
---

## Results Summary

| Metric             | Raspberry Pi 4 | Google Colab GPU |
|--------------------|----------------|------------------|
| Inference Time     | ~316 ms        | ~20.78 ms        |
| Accuracy           | ~98%           | ~98%             |
| Memory Footprint   | Low            | Higher           |
| Power Efficiency   | High           | Lower            |


---

## Future Scope

- Support for federated edge deployments
- Optimized GPU kernel with native CUDA
- Power profiling integration
- Use of real-time biosignal datasets (e.g., EMG, ECG)

---

## Applications

- Embedded AI for wearables
- Gesture recognition on edge
- Energy-efficient pattern recognition in IoT
- HDC experimentation for adaptive learning

---

## License

This project is for academic and research demonstration purposes only.  



