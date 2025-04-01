# Measuring the Overfitting of Cross-Validation in Radiomics

## Environment

Create a virtual environment:
`python3 -m venv ./venv`  

Activate it:  
`source ./venv/bin/activate`  

Install the required libraries:  
`pip install -r requirements.txt`

PymRMRe must be installed manually, as the version available via pip is outdated/broken.
Follow the instructions in the PymRMRe folder to install it.



## Datasets

The radMLBench datasets will be downloaded automatically and stored
in the `./datasets` folder. However, the UCI data must be retrieved manually
from the repository shared by Wainer and Gawley at:  
https://figshare.com/articles/dataset/Nested_cross_validation_is_overzealous/3457238  

After downloading, extract the datasets from the file `alldata.zip` into the `./dataUCI` folder.  
The resulting directory structure will be:

dataUCI/  
  ├── abalone/  
  │   ├── abalone.arff  
  ├── acute-inflammation/  
  │   ├── acute-inflammation.arff  
  ...  



## Experiment

Run the experiment with `python3 ./experiment.py --cohort=radMLBench` and
`python3 ./experiment.py --cohort=UCI`, for the radiomics and non-radiomics data respectively.

NOTE: Adjust the number of CPUs at the end of the script in the
line `results = Parallel(n_jobs=30)(`
Change  30 to match your system's setup.

**WARNING**: On a 16-core 2nd-gen Threadripper, the experiment took
nearly a month to complete. Newer CPUs might be faster, however,
keep in mind that it will take considerable CPU time.

**WARNING**: Overall around 400 GB of results will be created.
Ensure that your hard drive does not run out of space!


## Evaluation

Generate all figures and results in the `./paper` directory by running:
 `python3 ./evaluate.py`.

Evaluation can be performed without re-running the experiment, as the key results
(e.g., the best model and its predictions) are stored in `./paper/results_raw.dump`.
Other analysis can be performed, if the data needed happens to be stored in the results file.




## LICENSE

MIT License

Copyright (c) 2025 aydin demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
