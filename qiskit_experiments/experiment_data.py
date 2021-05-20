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
Experiment Data class
"""
import logging
from typing import Optional, Union, List, Dict, Tuple
import os
import uuid
from collections import OrderedDict

from qiskit.result import Result
from qiskit.providers import Job, BaseJob, JobStatus
from qiskit.providers.experiment import ExperimentDataV1
from qiskit.exceptions import QiskitError
from qiskit.providers import Job, BaseJob
from qiskit.providers.exceptions import JobError

from qiskit_experiments.matplotlib import pyplot, HAS_MATPLOTLIB


LOG = logging.getLogger(__name__)


class AnalysisResult(dict):
    """Placeholder class"""


class ExperimentData(ExperimentDataV1):
    """Qiskit Experiments Data container class"""

    def __init__(
        self,
        experiment: Optional["BaseExperiment"] = None,
        backend: Optional[Union["Backend", "BaseBackend"]] = None,
        job_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        share_level: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Initialize experiment data.

        Args:
            experiment: experiment object that generated the data.
            backend: Backend the experiment runs on. It can either be a
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
        # Experiment class object
        self._experiment = experiment

        backend = None
        if experiment is not None:
            experiment_type = experiment._type
        else:
            experiment_type = None

        super().__init__(
            backend,
            experiment_type=experiment_type,
            experiment_id=str(uuid.uuid4()),
            tags=tags,
            job_ids=job_ids,
            share_level=share_level,
            notes=notes,
        )

    @property
    def experiment(self):
        """Return Experiment object"""
        return self._experiment

    @property
    def experiment_type(self) -> str:
        """Return the experiment type."""
        return self._type

    @property
    def experiment_id(self) -> str:
        """Return the experiment id."""
        return self._id

    def status(self) -> str:
        """Return the data processing status.
        Returns:
            Data processing status.
        """
        if not self._jobs and not self._data:
            return "EMPTY"
        return super().status()

    def add_data(
        self,
        data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
    ):
        """Add experiment data.
        Args:
            data: Experiment data to add. Several types are accepted for convenience:

                * Result: Add data from this ``Result`` object.
                * List[Result]: Add data from the ``Result`` objects.
                * Job: Add data from the job result.
                * List[Job]: Add data from the job results.
                * Dict: Add this data.
                * List[Dict]: Add this list of data.

        Raises:
            QiskitError: if data format is invalid.
            KeyboardInterrupt: when job is cancelled by users.
        """
        # Set backend from the job, this could be added to base class
        if isinstance(data, (Job, BaseJob)):
            backend = data.backend()
            if self.backend is not None and str(self.backend) != str(backend):
                LOG.warning(
                    "Adding a job from a backend (%s) that is different than"
                    " the current ExperimentData backend (%s).",
                    backend,
                    self.backend,
                )
            self._backend = backend
            # Temporary hack to block until job is finished since collect function
            # doesn't seem to be working correctly
            data.result()
        super().add_data(data, post_processing_callback)

    def _add_result_data(self, result: Result) -> None:
        """Add data from a Result object

        Args:
            result: Result object containing data to be added.
        """
        num_data = len(result.results)
        for i in range(num_data):
            metadata = result.results[i].header.metadata
            if metadata.get("experiment_type") == self._type:
                data = result.data(i)
                data["metadata"] = metadata
                if "counts" in data:
                    # Format to Counts object rather than hex dict
                    data["counts"] = result.get_counts(i)
                self._add_single_data(data)

    def data(self, index: Optional[Union[int, slice, str]] = None) -> Union[Dict, List[Dict]]:
        """Return the experiment data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment data.

        Raises:
            QiskitError: if index is invalid.
        """
        # Get job results if missing experiment data.
        if self._jobs and not self._data and self._backend and self._backend.provider():
            for jid in self._jobs:
                if self._jobs[jid] is None:
                    try:
                        self._jobs[jid] = self._backend.provider().retrieve_job(jid)
                    except Exception:  # pylint: disable=broad-except
                        pass
                if self._jobs[jid] is not None:
                    self._add_result_data(self._jobs[jid].result())
        self._collect_from_queues()

        if index is None:
            return self._data
        if isinstance(index, (int, slice)):
            return self._data[index]
        if isinstance(index, str):
            return [data for data in self._data if data.get("job_id") == index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def add_figure(
        self,
        figure,
        figure_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Tuple[str, int]:
        """Save the experiment figure.

        Args:
            figure (Union[str, bytes, :class:`~matplotlib.figure.Figure`]): Name of the figure file
            or figure data to store. This can either be a ``str`` (for a filename to load),
            ``bytes`` (for the raw image data), or a :class:`~matplotlib.figure.Figure` object.
            figure_name: Name of the figure. If ``None``, use the figure file name, if
                given, or a generated name.
            overwrite: Whether to overwrite the figure if one already exists with
                the same name.

        Returns:
            A tuple of the name and size of the saved figure. Returned size
            is 0 if there is no experiment service to use.

        Raises:
            QiskitError: If the figure with the same name already exists,
                         and `overwrite=True` is not specified.
        """
        if not figure_name:
            if isinstance(figure, str):
                figure_name = figure
            else:
                figure_name = f"figure_{self.experiment_id}_{len(self.figure_names)}"

        existing_figure = figure_name in self._figure_names
        if existing_figure and not overwrite:
            raise QiskitError(
                f"A figure with the name {figure_name} for this experiment "
                f"already exists. Specify overwrite=True if you "
                f"want to overwrite it."
            )
        out = [figure_name, 0]
        self._figures[figure_name] = figure
        self._figure_names.append(figure_name)
        return out

    def figure(
        self, figure_name: Union[str, int], file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve the specified experiment figure.

        Args:
            figure_name: Name of the figure or figure position.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            Union[int, bytes, :class:`~matplotlib.figure.Figure`]:

                The size of the figure as an ``int`` if ``file_name`` is specified. Otherwise
                the content of the figure as ``bytes`` object or a
                :class:`~matplotlib.figure.Figure` depending on how the image was loaded.

        Raises:
            QiskitError: If the figure cannot be found.
        """
        if isinstance(figure_name, int):
            figure_name = self._figure_names[figure_name]

        figure_data = self._figures.get(figure_name, None)
        if figure_data is not None:
            if isinstance(figure_data, str):
                with open(figure_data, "rb") as file:
                    figure_data = file.read()
            if file_name:
                with open(file_name, "wb") as output:
                    if HAS_MATPLOTLIB and isinstance(figure_data, pyplot.Figure):
                        figure_data.savefig(output, format="svg")
                        num_bytes = os.path.getsize(file_name)
                    else:
                        num_bytes = output.write(figure_data)
                    return num_bytes
            return figure_data
        raise QiskitError(f"Figure {figure_name} not found.")

    def status(self) -> str:
        """Return the data processing status.

        Returns:
            Data processing status.
        """
        # TODO: Figure out what statuses should be returned including
        # execution and analysis status
        if not self._jobs and not self._data:
            return "EMPTY"
        return super().status()
