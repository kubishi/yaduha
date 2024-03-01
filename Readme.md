A Web App to create simple sentences in paiute.

# TODOs
- [ ] Package the app

# Usage
First, install the dependencies:
```bash
pip install -r requirements.txt
```

For OVP to English Translation:
```bash
python translate_ovp2eng.py --help
```

For English to OVP Translation:
```bash
python translate_eng2ovp.py --help
```
All results (if any) for running the above commands will be saved in the ```.results``` directory.

To plot the results from running ```python translate_eng2ovp.py evaluate```, run:
```bash
python plot_results.py
```
The results will be saved in the ```.output``` directory.
