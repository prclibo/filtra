# FILTRA: Rethinking Steerable CNN by Filter Transform

[![BSD-3-Clause](https://img.shields.io/github/license/prclibo/relative_pose)](https://github.com/prclibo/relative_pose/blob/master/LICENSE)
[![ECCV 2020](https://img.shields.io/badge/ICML-2021-%231b75bc)]()

This repository hosts the code for the FILTRA steerable CNN proposed in our ICML 2021 paper:

```
@inproceedings{DBLP:conf/icml/LiWL21,
  author    = {Bo Li and
               Qili Wang and
               Gim Hee Lee},
  editor    = {Marina Meila and
               Tong Zhang},
  title     = {{FILTRA:} Rethinking Steerable {CNN} by Filter Transform},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {6515--6522},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {http://proceedings.mlr.press/v139/li21v.html},
  timestamp = {Wed, 25 Aug 2021 17:11:17 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/LiWL21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

The convolution module has three APIs which can be found in `filtra/conv.py`. See the docstring inside for detailed usage.

Usually, to implement a full CNN for classification, you need convolution layers, activation layers and pooling layers. I am just too lazy and busy to implement activation and pooling layers so I wrapped FILTRA conv in the interface of E2CNN and re-use their other layers. But this results in runtime overhead to transpose tensor to the E2CNN order. It should not be difficult to implement FILTRA style activation and pooling layers if you really need it. *I will hopefully implement a version if I got time by July or later*.





