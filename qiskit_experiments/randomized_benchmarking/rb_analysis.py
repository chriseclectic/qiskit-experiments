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
Standard RB analysis class.
"""

import numpy as np
from scipy.optimize import curve_fit
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class RBAnalysis(BaseAnalysis):
    """Base Analysis class for analyzing Experiment data."""

    def _run_analysis(self, experiment_data, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            options: kwarg options for analysis function.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        # Initial guess
        xdata, ydata, num_qubits = self._format_data(experiment_data.data)
        fit_guess = self._fit_guess(xdata, ydata, num_qubits)
        bounds = ([0, 0, 0], [1, 1, 1])

        params, pcov = curve_fit(self._fit_fun, xdata, ydata, p0=fit_guess, bounds=bounds)
        params_err = np.sqrt(np.diag(pcov))

        scale = (2 ** num_qubits - 1) / (2 ** num_qubits)
        epc = scale * (1 - params[1])
        epc_err = scale * params_err[1] / params[1]
        analysis_result = {
            "EPC": epc,
            "EPC_err": epc_err,
            "params": params,
            "params_err": np.sqrt(np.diag(pcov)),
            "prams_cov": pcov,
        }
        ax = self._generate_plot(xdata, ydata, analysis_result)
        # TODO: figure out what to do with plots
        plt.show()
        return AnalysisResult(analysis_result), [ax]

    @staticmethod
    def _format_data(data):
        xdata = []
        ydata = []
        for datum in data:
            metadata = datum["metadata"]
            length = metadata["length"]
            num_qubits = len(metadata["qubits"])
            counts = datum["counts"]
            p0 = counts.get(num_qubits * "0", 0) / sum(counts.values())
            xdata.append(length)
            ydata.append(p0)
        return np.array(xdata), np.array(ydata), num_qubits

    @classmethod
    def _generate_plot(cls, xdata, ydata, analysis_result, ax=None, add_label=True):

        if not HAS_MATPLOTLIB:
            raise ImportError(
                "The function plot_rb_data needs matplotlib. "
                'Run "pip install matplotlib" before.'
            )

        if ax is None:
            plt.figure()
            ax = plt.gca()

        # Plot raw data
        ax.scatter(xdata, ydata, c="gray", marker="x")

        # Plot fit data
        xfit = np.linspace(min(xdata), max(xdata), 100)
        yfit = [cls._fit_fun(x, *analysis_result["params"]) for x in xfit]
        ax.plot(xfit, yfit, color="blue", linestyle="-", linewidth=2)

        # Plot mean and std dev
        xvals = np.unique(xdata)
        ymean = np.zeros(xvals.size)
        yerr = np.zeros(xvals.size)
        for i in range(xvals.size):
            ys = ydata[xdata == xvals[i]]
            ymean[i] = np.mean(ys)
            yerr[i] = np.std(ys)
        plt.errorbar(xvals, ymean, yerr=yerr, color="r", linestyle="--", linewidth=3)

        # Formatting
        ax.tick_params(labelsize=14)

        ax.set_xlabel("Clifford Length", fontsize=16)
        ax.set_ylabel("Ground State Population", fontsize=16)
        ax.grid(True)

        if add_label:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2)

            ax.text(
                0.6,
                0.9,
                "alpha:{:.3f}({:.3f}) EPC: {:.3f}({:.3f})".format(
                    analysis_result["params"][1],
                    analysis_result["params_err"][1],
                    analysis_result["EPC"],
                    analysis_result["EPC_err"],
                ),
                ha="center",
                va="center",
                size=14,
                bbox=bbox_props,
                transform=ax.transAxes,
            )
        return ax

    @staticmethod
    def _fit_fun(x, a, alpha, b):
        """Function used to fit RB."""
        # pylint: disable=invalid-name
        return a * alpha ** x + b

    @staticmethod
    def _fit_guess(xdata, ydata, num_qubits):
        xmin = np.min(xdata)
        y_mean_min = np.mean(ydata[xdata == xmin])

        xmax = np.max(xdata)
        y_mean_max = np.mean(ydata[xdata == xmax])

        b_guess = 1 / (2 ** num_qubits)
        a_guess = 1 - b_guess
        alpha_guess = np.exp(
            np.log((y_mean_min - b_guess) / (y_mean_max - b_guess)) / (xmin - xmax)
        )
        return [a_guess, alpha_guess, b_guess]
