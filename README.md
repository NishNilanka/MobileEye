# 📱 MobileEYE: Deep Learning-based Mobile Device Eye Tracking

**MobileEYE** is a deep learning-based gaze estimation framework designed to work with the **front-facing camera of mobile devices**, enabling eye-tracking capabilities without the need for expensive external hardware. This repository includes source code, trained models, and evaluation scripts used in the paper:

> Gunawardena, N., Ginige, J. A., Javadi, B., & Lui, G. (2024).  
> **Deep learning based eye tracking on smartphones for dynamic visual stimuli**.  
> *Procedia Computer Science*, 246, 3733–3742. Elsevier.  
> [https://doi.org/10.1016/j.procs.2024.09.183](https://doi.org/10.1016/j.procs.2024.09.183)

---

## 🔍 Overview

MobileEYE focuses on appearance-based eye tracking for **dynamic visual stimuli** such as videos and games. It leverages **CNN, CNN+GRU, and CNN+LSTM** architectures trained on two custom-built datasets collected on smartphones in real-world settings.

---

## 📂 Repository Structure

```
MobileEye/
│
├── Code/                     # Source code for data preprocessing, training, evaluation
├── Figures/                  # Diagrams and visualizations used in the paper
├── Models/
│   └── Original_models_tf/   # Pre-trained TensorFlow models (CNN, CNN+GRU, CNN+LSTM)
├── Videos/                   # Demo video of the eye-tracking system
├── README.md                 # Project overview and instructions
```

---

## 🚀 Features

- 📷 Gaze estimation using front camera only
- 🧠 Supports CNN, CNN+GRU, and CNN+LSTM architectures
- 📊 Evaluation on multiple edge devices: Jetson, RPi, Intel NUC, Odroid
- 🔎 Sensitivity analysis: age, gender, lighting, device type, etc.
- 📈 Benchmark metrics: RMSE, R², inference time, energy, CPU usage

---

## 📦 Installation

```bash
git clone https://github.com/NishNilanka/MobileEye.git
cd MobileEye
pip install -r requirements.txt
```

## 📝 Citation

If you use this work in your research, please cite the following:

```bibtex
@article{gunawardena2024deep,
  title={Deep learning based eye tracking on smartphones for dynamic visual stimuli},
  author={Gunawardena, Nishan and Ginige, Jeewani Anupama and Javadi, Bahman and Lui, Gough},
  journal={Procedia Computer Science},
  volume={246},
  pages={3733--3742},
  year={2024},
  publisher={Elsevier}
}
```

## 📬 Contact

For questions or collaborations, feel free to raise an issue or contact me via [n.gunawardena@westernsydney.edu.au](mailto:n.gunawardena@westernsydney.edu.au).
