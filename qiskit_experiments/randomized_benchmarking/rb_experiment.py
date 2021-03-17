# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Standard RB Experiment class.
"""
from typing import Union, Iterable, Optional

import numpy as np
from numpy.random import Generator, default_rng

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, random_clifford

from qiskit_experiments.base_experiment import BaseExperiment
from .rb_analysis import RBAnalysis


class RBExperiment(BaseExperiment):
    """RB Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = RBAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]],
        lengths: Iterable[int],
        num_samples: int = 1,
        seed: Optional[Union[int, Generator]] = None,
    ):
        """Standard randomized benchmarking experiment

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            num_samples: number of samples to generate for each
                         sequence length
            seed: Seed or generator object for random number
                  generation. If None default_rng will be used.
        """
        if not isinstance(seed, Generator):
            self._rng = default_rng(seed=seed)
        else:
            self._rng = seed
        self._lengths = list(lengths)
        self._num_samples = num_samples

        super().__init__(qubits)

    # pylint: disable = arguments-differ
    def circuits(self, backend=None):
        """Return a list of RB circuits.

        Args:
            backend (Backend): Optional, a backend object.

        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.
        """
        circuits = []
        for length in self._lengths:
            for _ in range(self._num_samples):
                circ = self._sample_circuit(length, seed=self._rng)
                circ.measure_all()
                circuits.append(circ)
        return circuits

    def transpiled_circuits(self, backend=None, **kwargs):
        """Return a list of transpiled RB circuits.

        Args:
            backend (Backend): Optional, a backend object to use as the
                               argument for the :func:`qiskit.transpile`
                               function.
            kwargs: kwarg options for the :func:`qiskit.transpile` function.

        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.

        Raises:
            QiskitError: if an initial layout is specified in the
                         kwarg options for transpilation. The initial
                         layout must be generated from the experiment.
        """
        circuits = super().transpiled_circuits(backend=backend, **kwargs)
        # TODO: Add functionality for gates per clifford which depends
        # on the transpiled circuit gates
        return circuits

    def _sample_circuit(self, length, seed=None):
        """Sample a single RB circuits"""
        qubits = list(range(self.num_qubits))
        metadata = {
            "experiment_type": self._type,
            "xdata": length,
            "group": "Clifford",
            "qubits": self.physical_qubits,
        }

        circ_op = Clifford(np.eye(2 * self.num_qubits))
        circ = QuantumCircuit(self.num_qubits)
        circ.metadata = metadata
        circ.barrier(qubits)

        # Add random group elements
        for _ in range(length):
            group_elt = random_clifford(self.num_qubits, seed=seed)
            circ_op = circ_op.compose(group_elt)
            circ.append(group_elt, qubits)
            circ.barrier(qubits)

        # Add inverse
        inv = circ_op.adjoint()
        circ.append(inv, qubits)
        circ.barrier(qubits)

        return circ
