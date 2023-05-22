## Description

This repository is for paper:  Can Code Representation Boost Information-Retrieval-Based Test Case Prioritization?

Test case prioritization (TCP) aims to schedule the execution order of test cases for earlier fault detection. 
A recent study has demonstrated that the information-retrieval-based (IR-based) TCP approaches achieve the state-of-the-art effectiveness. 
The current IR-based TCP approaches leverage lexical similarity between test cases and code changes to guide TCP while ignoring rich code semantics, which may limit their effectiveness to some degree.
In this paper, we conduct the first study to explore whether code semantic information can further boost IR-based TCP. 
Here, we studied two types of code representation methods (i.e., general-purpose and task-associated models) and explored two modes of utilizing the code representation embeddings, (i.e., unsupervised and supervised modes) for IR-based TCP. 
Our results demonstrate that incorporating code semantics through the supervised mode of code representation can achieve 16.96% improvement in the efficiency of fault detection over the state-of-the-art IR-based TCP approach (which is based on lexical similarity).

## The Structure

Here, we briefly introduce the usage/function of each directory: 

In this paper, we totally explore three RQs, 

**RQ1 : Is the semantic similarity between code changes and bug-triggering test cases higher than that between code changes and non-bug-triggering test cases?**

**RQ2: Can semantic code representation improve IR-based TCP?**

**RQ3: Is it possible to further improve the semantic-based TCP?**

- `astnn`:  Files for RQ1 and RQ2 of ASTNN
- `doc2vec`: Files for RQ1 and RQ2 of Doc2Vec
- `approach`: Files for RQ3 of our approach **SSTcp** ,  an IR-based TCP approach with code representation.
- `data`: Data used for evaluating our approach **SSTcp**. Specifically, we provide D237, which is collcted by us as benchmark.

## Reproducibility

RQ1 and RQ2 are placed in two directories `astnn` and `doc2vec`.

In  `astnn` :

to evaluate RQ1,  we can execute relative orders in path `astnn`

~~~shell
python process.py # preprocess test cases
python embedding.py # encode test cases into code embeddings and similarity
python pipeline.py # check if each test case can be parsed into AST
~~~

to evaluate RQ2, we can execute get_apfd.py,  please note that to evaluate RQ2 requires execution of instruction above.

~~~shell
python get_apfd.py # calculate APFD of ASTNN (for RQ2)
~~~

In  `doc2vec` :

to evaluate RQ1:

~~~shell
python main.py --epochs 10 --model doc2vec --fname sim.json --job_dir 12345 
~~~

`--epochs` inference epochs

`--model` refers to the model selected for code representation

`--fname` refers to the json file named to be saved (containing similarities between code fragments)

`--job_dir` refers to directory containing jobs

to evaluate RQ2, we can execute get_apfd.py,  please note that to evaluate RQ2 requires execution of instruction above.

~~~shell
python get_apfd.py # calculate APFD of ASTNN (for RQ2)
~~~

In  `approach` :

programs related to RQ3 are listed in path `approach` :

In this RQ, we mainly evaluate the effectiveness of supersized fine-tuning on downstream tasks on IR-based TCP. 

The main instructions for RQ3 are the three files below: run_make_data.py,run_test.py,run_train.py; To replicate the experimental results, you can directly run instructions below:

~~~shell
python run_make_data.py # This file aims to construct data split for downstream training
python run_train.py # fine-tune the model on downstream dataset via supervised learning
python run_test.py # evaluate the performance of fine-tuned model 
~~~

The testing results will be saved in 'test_results' directory and information related to fine-tuning will be saved in 'logs' directory. 

## Requirements

- python 3.6
- pandas 0.20.3
- gensim 3.5.0
- scikit-learn 0.19.1
- pytorch 1.0.0
- pycparser 2.18
- javalang 0.11.0
- numpy 1.14.2

## Data

All the data about our new collected benchmark 'D237' has been uploaded in `data` for further reproduction.
