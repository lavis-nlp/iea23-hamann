# Domain-Specific Knowledge Graph Adaption with Industrial Text Data

Original code used for the system presented on the IEA/AIE 2023
(IKEDMS). As we unfortunately cannot provide the original training
data, this code can be seen as documentation to better understand the
paper.

- We offer the scattered phrase matcher as a standalone spaCy component here: [lavis-nlp/scaphra](https://github.com/lavis-nlp/scaphra)
- This model is the precursor to the e2e models of IRT1 and IRT2: Check out the latest model implementation here: [lavis-nlp/irt2m](https://github.com/lavis-nlp/irt2m)
- The training configuration of the models presented in the paper can be found in
  - `conf/models/symptax.v7/kgc3-ce-1.yml`
  - `conf/models/symptax.v7/kgc3-ns-1.yml`


Entry points:

- Used model: [CE: draug/models/models.py#L869](https://github.com/lavis-nlp/iea23-hamann/blob/main/draug/models/models.py#L869)
- Used model: [NS: draug/models/models.py#L919](https://github.com/lavis-nlp/iea23-hamann/blob/main/draug/models/models.py#L919)
- Training data sampling strategies: [draug/models/data.py](https://github.com/lavis-nlp/iea23-hamann/blob/main/draug/models/data.py)
- Scattered text sampling: [draug/homag/sampling.py#L143](https://github.com/lavis-nlp/iea23-hamann/blob/main/draug/homag/sampling.py#L143)


If you find this work interesting, please consider a citation

```
Coming soon
```
