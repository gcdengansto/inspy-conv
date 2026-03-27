# inspy-conv

**inspy-conv**, renamed from **inspy**, is a Python toolkit for scientific data analysis and simulation workflows, with a focus on inelastic neutron scattering instrument resolution calculation and convolution data fitting.

> Install name: `inspy-conv`  
> Import name: `inspy`

---

## 🚀 Installation

```bash
pip install inspy-conv
```
> ⚠️ **WARNING**: "pip install inspy" will install another package, which has nothing to do with inelastic neutron scattering. 

## 📦 Usage
**REMINDer**: After "pip install inspy-conv", one can use this package just using "import inspy", see below:

```bash
import inspy

# Example usage
# (replace with real functionality)
result = inspy.do_something()
print(result)
```
Run the GUI, Go to terminal:
```bash
>python
>>import inspy
>>inspy.gui.main_gui.main()
or
>>inspy.gui.gui_convfit_escan_uf.main()
>>
```
On linux like Ubuntu and Linuxmint, if you have warning like this:
```bash
QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-user'libGL error: glx: failed to create dri3 screenlibGL error: failed to load driver: virtio_gpu python
```
add the following lines in your .bashrc
```bash
# Ensure always render the GUI using software
export LIBGL_ALWAYS_SOFTWARE=1

# Ensure XDG_RUNTIME_DIR exists for third-party apps
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR=/tmp/runtime-$USER
    if [ ! -d "$XDG_RUNTIME_DIR" ]; then
        mkdir -p "$XDG_RUNTIME_DIR"
        chmod 0700 "$XDG_RUNTIME_DIR"
    fi
fi
```
then run
```bash
source ~/.bashrc
```
Then, you can rerun the GUI interface in python again.


## 🎯 Features
Modular design for scientific workflows
Lightweight and easy to integrate into existing pipelines
Designed for extensibility in research environments
Compatible with NumPy-based data processing


## 📁 Project Structure
```bash
inspy-conv/
├── src/inspy/        # Main package (import inspy)
├── pyproject.toml    # Build configuration
├── README.md
└── LICENSE
```

## 🔬 Scope

This package is intended for:

> Neutron Scattering Data Analysis

> Instrument Resolution Calculation

> Resolution Convolution Fitting to Inelastic Neutron Scattering Data

> General Inelastic Neutron Scattering Geometric Calculation

## 🛠️ Development

Repository:

https://github.com/gcdengansto/inspy.git


Install in editable mode:
```bash
pip install -e .
```

## 📖 Documentation

Documentation is under development.
Usage examples and tutorials can be found at https://github.com/gcdengansto/inspy.

## 🤝 Contributing

Please contact the author for fixing bugs and adding additional functions


## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 👤 Author

Guochu Deng

Email: gc.deng.ansto@gmail.com

## ⚠️ Disclaimer

This software is provided for scientific research purposes.
No guarantees are made regarding correctness or fitness for a particular application.
