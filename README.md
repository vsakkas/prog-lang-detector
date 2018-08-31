# Programming Language Detector
A programming language detector written in Python using [Scikit-Learn](https://github.com/scikit-learn/scikit-learn). Uses a [Random Forest Classifier](https://en.wikipedia.org/wiki/Random_forest) and it was trained to correctly identify a total of 50 programming languages, which are the following: `Ada`, `AppleScript`, `AWK`, `BBC BASIC`, `C`, `C++`, `C#`, `Clojure`, `COBOL`, `Common Lisp`, `D`, `Elixir`, `Erlang`, `Forth`, `Fortran`, `Go`, `Groovy`, `Haskell`, `Icon`, `J`, `Java`, `JavaScript`, `Julia`, `Kotlin`, `LiveCode`, `Lua`, `Maple`, `MATLAB`, `Objective C`, `OCaml`, `Oz`, `Perl`, `PHP`, `PL-I`, `PowerShell`, `Prolog`, `Python`, `R`, `Racket`, `REXX`, `Ring`, `Ruby`, `Rust`, `Scala`, `Scheme`, `Swift`, `Tcl`, `UNIX Shell`, `VBScript` and `Visual-Basic .NET`.

In order to have a large enough dataset for the above languages, the [Roseta Code Dataset](https://github.com/acmeism/RosettaCodeData) was used for training. Below are some metrics that were produced with 10-Fold Cross Validation in order to determine the performance of the trained classifier:

| Accuracy | Precision | Recall  | F1   |
| :-------:|:---------:|:-------:|:----:|
| 93.93%   | 94.77%    | 92.75%  |93.51%|

*Note: in order to produce the above results, `80%` of the dataset was used training and the other `20%` for calculating the performance of the classifier.*

## Getting Started

To get the code up and running on your local machine, simply follow the following instructions.

### Prerequisites

First, you need to download `scikit-learn` (version 0.19 and newer) using the following command:

```
pip3 install -U scikit-learn
```

*Note: make sure `python3` (version 3.5 and newer) is installed.*

### Downloading

To download the source code of this project use the following command:

```
git clone https://github.com/vsakkas/prog-lang-detector.git
```

And to enter the directory of the downloaded project, simply type:
```
cd prog-lang-detector
```

### Running

To train the classifier, simply run the following command:

```
python3 src/prog_lang_detector.py --train <dataset>
```
In the above command, `<dataset>` needs to be folder. The provided argument must end with a `/` and it must contain at least 50 directories, one for each of the languages to be used for training.

*Note: running the above command will generate the following pickle files: `dataset.pkl`, `tfidf.pkl`, `nmf.pkl`, `train.pkl`,  `test.pkl`, `classifier.pkl`. The last file contains the trained classifier. This file along with `tfidf.pkl` and `nmf.pkl` are required in order for the `--predict` command to work.*
 
Finally, to use the trained classifier to predict what language a file us, use the following command:

```
python3 src/prog_lang_detector.py --predict <file>
```

The command above will simply print the predicted language for the given file.

*Note: the extension of the provided file is not taken under consideration when trying to predict what programming language the code of the file is. Only the file's content is used.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


