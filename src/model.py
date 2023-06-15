from __future__ import annotations
import collections 
from dataclasses import dataclass, asdict
import datetime
import sys
from typing import Optional,List,Counter

import weakref 


@dataclass(frozen=True)
class Sample: #데이터 학습? 데이터 셋 저장
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@dataclass(frozen=True)
class KnownSample(Sample):
    species: str


@dataclass(frozen=True)
class TrainingKnownsample:
    sample: KnownSample


@dataclass(frozen=True)
class TestingKnownsample: 
    sample: KnownSample
    classification: Optional[str] = None

 
@dataclass(frozen=True)
class UnknownSample:
    """Sample provided by user, net yet classified"""

    sample: Sample
    classification: Optional[str] = None 


class Distance:
    def distance(self, s1: Sample, s2: Sample) -> float:
        raise NotImplementedError

@dataclass
class Hyperparameter: #학습하는 .. 므ㅏㄹ파라미터? 학습 하이퍼? 응?

    k: int
    algorithm: Distance
    data: weakref.ReferenceType["TrainingData"]

    def classify(self, sample: Sample) -> str:#진짜학습
        """k-NN algorithm"""
       
        if not ( training_data := self.data()):
            raise RuntimeError("No traninngData object!")
        distances: list[tuple[float,TrainingKnownsample]] = sorted(
                (self.algorithm.distance(sample, Known.sample), Known)
                for Known in training_data.training
        )
        k_nearest = (known.sample.species for _, known in distances[:self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()  # ("a",5)
        species, votes = best_fit
        return species

@dataclass
class TrainingData:
    testing: List[TestingKnownsample]
    Training: List[TrainingKnownsample]
    tuning: List[Hyperparameter]
        

test_sample ="""
>>> x = Sample(1.0, 2.0, 3.0, 4.0)
>>> x 
UnKnownSample(sepal_length = 1.0, sepal_width = 2.0, petal_lenhth = 3.0, petal_width = 4.0, species = None)

"""

test_TrainingKnownSample = """
>>> s1 = TrainingKnownSample(
...     sample = KnownSample(
...         sepal_lengt = 5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species = "Iris-setosa"
...     )
... )
>>> s1
TrainingKnownSample(sample = KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species = "Iris-setosa"

#This is preferrable...

>>> s1. classification = "Wrong"
Trackback (most recent call last):
...
datacalsses.FrozenInstanceError: cannot assign to field 'classification'

>>> hash(s1) is not None
True
"""

test_TestingKnownSample = """
>>> s2 = testingKnownSample(
...     KnownSample(sepal_length = 5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, species = "Iris-setosa"),
...     classification = None
... )
>>> s2
TestingKnownSample(sample = KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=:1.4, petal_width=0.2, species='Iris-setosa), Classification=None)

#This is more expected...

>>> s2. classification = "Wrong"
>>> s2
TestingKnownSample(sample = KnownSample(sepal_length=5.1, sepal_width=3.5, petal_length=:1.4, petal_width=0.2, species='Iris-setosa), Classofocation='wrong'

datacalsses.FrozenInstanceError: cannot assign to field 'classification'

"""


test_UnkownSample = """
>>> u = UnknownSample(
...     sample = Sample(sepal_length=5.1, sepal_width=3.5, petal_length=:1.4, petal_width=0.2),
...     classification=None)
>>> u
UnKnownSample(sample=Sample(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2), classification=None)
)

"""
__test__ = {
    name: case for name, case in globals().items() if name.startswith("test_")
    }
