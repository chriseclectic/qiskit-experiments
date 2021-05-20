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
Composite Experiment data class.
"""

from typing import Optional, Union, List
from qiskit.result import marginal_counts
from qiskit.exceptions import QiskitError
from qiskit_experiments.experiment_data import ExperimentData


class CompositeExperimentData(ExperimentData):
    """Composite experiment data class"""

    def __init__(
        self,
        experiment: "CompositeExperiment",
        backend: Optional[Union["Backend", "BaseBackend"]] = None,
        job_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        share_level: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Initialize experiment data.

        Args:
            experiment (CompositeExperiment): experiment object that generated the data.
            backend (Backend): Optional, Backend the experiment runs on. It can either be a
                :class:`~qiskit.providers.Backend` instance or just backend name.
            job_ids: IDs of jobs submitted for the experiment.
            tags: Tags to be associated with the experiment.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the experiment service supports sharing. See
                the specific service provider's documentation on valid values.
            notes: Freeform notes about the experiment.

        Raises:
            ExperimentError: If an input argument is invalid.
        """

        super().__init__(
            experiment,
            backend=backend,
            tags=tags,
            job_ids=job_ids,
            share_level=share_level,
            notes=notes,
        )

        # Initialize sub experiments
        self._components = [
            expr.__experiment_data__(expr, backend=backend)
            for expr in experiment._experiments]

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nStatus: {status}"
        ret += f"\nComponent Experiments: {len(self._components)}"
        ret += f"\nCircuits: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result"
            for key, value in self._analysis_results[-1].items():
                ret += f"\n- {key}: {value}"
        return ret

    def component_experiment_data(
        self, index: Optional[Union[int, slice]] = None
    ) -> Union[ExperimentData, List[ExperimentData]]:
        """Return component experiment data"""
        if index is None:
            return self._components
        if isinstance(index, (int, slice)):
            return self._components[index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def _add_single_data(self, data):
        """Add data to the experiment"""
        # TODO: Handle optional marginalizing IQ data
        metadata = data.get("metadata", {})
        if metadata.get("experiment_type") == self._type:

            # Add parallel data
            self._data.append(data)

            # Add marginalized data to sub experiments
            if "composite_clbits" in metadata:
                composite_clbits = metadata["composite_clbits"]
            else:
                composite_clbits = None
            for i, index in enumerate(metadata["composite_index"]):
                sub_data = {"metadata": metadata["composite_metadata"][i]}
                if "counts" in data:
                    if composite_clbits is not None:
                        sub_data["counts"] = marginal_counts(data["counts"], composite_clbits[i])
                    else:
                        sub_data["counts"] = data["counts"]
                self._components[index].add_data(sub_data)
