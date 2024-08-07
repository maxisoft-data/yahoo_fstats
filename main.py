import concurrent.futures
import contextlib
import copy
import os
import random
import sys
import time
import warnings
from pathlib import Path
from threading import Event
from typing import Optional

import fathon
import numpy as np
import pandas as pd
import statsmodels.api as sm
import stumpy
from PyEMD import EMD
from numpy.lib.stride_tricks import sliding_window_view
from pathvalidate import sanitize_filename
from tqdm.auto import tqdm, trange


def check_env():
    assert Path('../input').is_dir()


def list_files():
    def gen():
        yield from Path('../input').rglob('*.csv')
        yield from Path('../input').rglob('*.csv.xz')

    return sorted(set(gen()), key=lambda p: p.name.lower())


window_choice = [32, 64, 90, 128, 240, 512, 1024, 2048]

trange_disabled = os.getenv('TRANGE_DISABLED', '').lower() in ('true', '1', 'y', 'yes', 'ok')

_random_generator = np.random.default_rng()


def rand_range_8(length: int):
    res: np.ndarray = np.arange(length)
    if res.shape[0] % 8 == 0:
        res = res.reshape(-1, 8)
        _random_generator.shuffle(res, axis=0)
    else:
        pt = res.shape[0] // 8 * 8
        tmp = res[:pt].reshape(-1, 8)
        _random_generator.shuffle(tmp, axis=0)
        tmp = tmp.flatten()
        tmp2 = res[pt:]
        _random_generator.shuffle(tmp2)
        ptr = np.random.randint(pt + 1)
        if ptr >= pt:
            res = np.concatenate((tmp.flatten(), tmp2))
        else:
            res = np.concatenate((tmp[:ptr], tmp2, tmp[ptr:]))

    return res.flatten()


def compute_price(ohlcav: pd.DataFrame, testing=True) -> np.ndarray:
    hl = np.where(
        ohlcav['close'].values > ohlcav['open'].values,
        ohlcav['high'].values,
        ohlcav['low'].values
    )

    hl2 = np.where(
        np.abs(ohlcav['close'].values - ohlcav['high'].values) > np.abs(ohlcav['close'].values - ohlcav['low'].values),
        ohlcav['low'].values,
        ohlcav['high'].values
    )

    res = np.float64(5) * np.ascontiguousarray(ohlcav['close'].values, dtype=np.float64)
    res += np.where(np.isfinite(ohlcav['adj close'].values), ohlcav['adj close'].values, ohlcav['close'].values)
    res += hl
    res += hl2
    res += ohlcav['high'].values
    res += ohlcav['low'].values

    res /= 5 + 5

    if testing:
        np.testing.assert_allclose(np.log2(res), np.log2(ohlcav['close'].values), rtol=1e-1)
    return res


def _pickup_sub_range(ohlcav: pd.DataFrame, max_mult=5) -> Optional[pd.DataFrame]:
    window_max = max(window_choice)
    length = np.random.randint(window_max * 3, window_max * max_mult) // 16 * 16

    if len(ohlcav) < length:
        return None
    start_index = 0

    while start_index / len(ohlcav) < np.random.random():
        start_index = np.random.randint(0, len(ohlcav) - length)

    return ohlcav.iloc[start_index:start_index + length].copy()


def compute_log_price(price: np.ndarray) -> np.ndarray:
    res = np.log10(price)
    if np.all(res > 0):
        return res
    res = np.log(price)
    if np.all(res > 0):
        return res
    res = np.sqrt(price)
    if np.all(res > 0):
        return res
    return price


def compute_auto_correlation(price: np.ndarray, window: int, out: np.ndarray,
                             normalize: bool = True, cancellation_event: Optional[Event] = None,
                             desc="Computing ACF") -> np.ndarray:
    """
    Computes the autocorrelation (ACF) on a rolling window basis.

    Args:
        price: The time series data as a NumPy array.
        window: The window size for the detrending procedure.
        out: The output array to store the slope of the linear trend.
        normalize: (Optional) Whether to normalize the data before ACF calculation. Defaults to True.
        cancellation_event: (Optional) An event object to monitor for cancellation requests.
        desc: (Optional) Description string for the progress bar. Defaults to "Computing ACF".

    Returns:
        The modified output array containing the slope of the linear trend for each window.
    """

    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]

    # Use tqdm.trange for progress bar with description
    for i in trange(len(view), desc=desc, leave=False, disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break

        sub = view[i, :]
        if normalize:
            sub = sub - np.mean(sub)
            std = np.std(sub)
            if np.isfinite(std) and std > 0:
                sub /= std

        try:
            acf = sm.tsa.stattools.acf(sub, adjusted=True)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute acf %s" % e, file=sys.stderr)
            continue  # Skip to the next iteration on error

        try:
            A = np.ones((acf.shape[0], 2), dtype=np.float32)
            A[:, 0] = np.arange(A.shape[0])
            solution, _ = np.linalg.lstsq(A, acf, rcond=-1)[0]
            out[i] = solution
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass  # Ignore errors during linear regression fit

    return out


def compute_partial_auto_correlation(price: np.ndarray, window: int, out: np.ndarray,
                                     normalize: bool = True, cancellation_event: Optional[Event] = None,
                                     desc="Computing PACF") -> np.ndarray:
    """
    Computes the partial autocorrelation (PACF) on a rolling window basis.

    Args:
        price: The time series data as a NumPy array.
        window: The window size for the detrending procedure.
        out: The output array to store the results (slope, residual, and Chebyshev fit coefficient).
        normalize: (Optional) Whether to normalize the data before PACF calculation. Defaults to True.
        cancellation_event: (Optional) An event object to monitor for cancellation requests.
        desc: (Optional) Description string for the progress bar. Defaults to "Computing PACF".

    Returns:
        The modified output array containing PACF results.
    """

    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]

    # Use tqdm.trange for progress bar with description
    for i in tqdm(rand_range_8(len(view)), desc=desc, leave=False, disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break

        sub = view[i, :]
        if normalize:
            sub = sub - np.mean(sub)
            std = np.std(sub)
            if np.isfinite(std) and std > 0:
                sub /= np.std(sub)

        try:
            pacf = sm.tsa.stattools.pacf(sub, method="ldadjusted")
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute pacf %s" % e, file=sys.stderr)
            continue  # Skip to the next iteration on error

        try:
            A = np.ones((pacf.shape[0], 2), dtype=np.float32)
            A[:, 0] = np.arange(A.shape[0])
            solution, residual = np.linalg.lstsq(A, pacf, rcond=-1)[0]
            out[i, 0] = solution
            out[i, 1] = residual
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass  # Ignore errors during linear regression fit

        try:
            coef = np.polynomial.chebyshev.chebfit(np.arange(len(pacf)), pacf, 2)
            out[i, 2] = coef[0]
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass  # Ignore errors during Chebyshev fit

    return out


def compute_matrix_profile_snippet(price: np.ndarray, window: int, out: np.ndarray, normalize=True) -> np.ndarray:
    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]
    snippet_window = np.sqrt(window)
    snippet_window = int(max(snippet_window, 3))
    for i in range(len(view)):
        try:
            series = view[i, :]
            if normalize:
                series = (series - np.mean(series)) / np.std(series, ddof=1)
            snippets, snippets_indices, mp, *_ = stumpy.snippets(series, m=snippet_window, k=1, normalize=False)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute snippet %s" % e, file=sys.stderr)
        else:
            snippets = np.squeeze(snippets)
            coef = np.polynomial.hermite.hermfit(np.arange(len(snippets)), snippets, 2)
            out[i] = coef[0]
    return out


def compute_mdfa(price: np.ndarray, window: int, out: np.ndarray, q=None, order=2,
                 cancellation_event: Optional[Event] = None) -> np.ndarray:
    """
    Computes the Multifractal Detrended Fluctuation Analysis (MFDFA) of a time series.

    Args:
        price: The time series data as a NumPy array.
        window: The window size for the detrending procedure.
        out: The output array to store the results.
        q: (Optional) An array of scaling exponents. If None, a default set of exponents will be used.
        order: (Optional) The polynomial order for fitting the fluctuation function.
        cancellation_event: (Optional) An event object to monitor for cancellation requests.

    Returns:
        The modified output array containing the Hurst exponent (H), scaling exponent (tau),
        and singularity spectrum (spectrum) estimates.
    """

    if q is None:
        dyna_sqrt = np.sqrt(window)
        while dyna_sqrt > np.log2(window):
            dyna_sqrt = np.sqrt(dyna_sqrt)
        q = np.asfarray((-dyna_sqrt, -2 * np.pi, (1 - np.sqrt(5)) / 2, np.sqrt(2) / 2, np.pi, np.exp(np.sqrt(2)) + 1))
        q = np.sort(q)

    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]

    olag = np.unique(np.logspace(2, 10, window, base=2).astype(np.int64))
    olag = olag[olag < window // 2]

    for i in tqdm(rand_range_8(len(view)), leave=False, desc="Computing mdfa", disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break

        try:
            pymfdfa = fathon.MFDFA(view[i, :])
            pymfdfa.computeFlucVec(olag, qList=q, polOrd=order, revSeg=True)
            list_H, list_H_intercept = pymfdfa.fitFlucVec()
            tau = pymfdfa.computeMassExponents()
            singularity, spectrum = pymfdfa.computeMultifractalSpectrum()

            H_p, H_residual, *_ = np.polynomial.polynomial.polyfit(q, list_H, 1, full=True)
            tau_p = np.polynomial.hermite.hermfit(q, tau, 2)
            # spectrum_fit = np.polynomial.hermite.hermfit(np.arange(len(spectrum)), spectrum, 2)

        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute mdfa %s" % e, file=sys.stderr)
        else:
            out[i, 0] = H_p[0]
            out[i, 1] = tau_p[0]
            out[i, 2:] = spectrum

    return out


def compute_dfa(price: np.ndarray, window: int, out: np.ndarray, order=2,
                cancellation_event: Optional[Event] = None) -> np.ndarray:
    """
    Computes the Detrended Fluctuation Analysis (DFA) of a time series.

    Args:
        price: The time series data as a NumPy array.
        window: The window size for the detrending procedure.
        out: The output array to store the results.
        order: (Optional) The polynomial order for fitting the fluctuation function.
        cancellation_event: (Optional) An event object to monitor for cancellation requests.

    Returns:
        The modified output array containing the Hurst exponent (H) estimates.
    """

    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]
    olag = np.unique(np.logspace(2, 10, window, base=2).astype(np.int64))
    olag = olag[olag < window // 2]

    for i in tqdm(rand_range_8(len(view)), leave=False, desc="Computing dfa", disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break

        try:
            dfa = fathon.DFA(view[i, :])
            dfa.computeFlucVec(olag, polOrd=order, revSeg=True, unbiased=view.shape[1] < 128)
            list_H, list_H_intercept = dfa.fitFlucVec()
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute dfa %s" % e, file=sys.stderr)
        else:
            out[i] = list_H

    return out


def compute_hurst(price: np.ndarray, window: int, out: np.ndarray, order=1) -> np.ndarray:
    h = fathon.HT(price)
    list_H = h.computeHt(window, mfdfaPolOrd=order, polOrd=order)
    out[-list_H.shape[1]:] = np.squeeze(list_H, axis=0)
    return out


def compute_adf(price: np.ndarray, window: int, out: np.ndarray,
                cancellation_event: Optional[Event] = None, desc="Computing ADF") -> np.ndarray:
    """
    Computes the Augmented Dickey-Fuller (ADF) test for stationarity on a rolling window basis.

    Args:
        price: The time series data as a NumPy array.
        window: The window size for the detrending procedure.
        out: The output array to store the p-values.
        cancellation_event: (Optional) An event object to monitor for cancellation requests.
        desc: (Optional) Description string for the progress bar. Defaults to "Computing ADF".

    Returns:
        The modified output array containing the p-values from the ADF test.
    """

    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]

    # Use tqdm.trange for progress bar with description
    for i in trange(len(view), desc=desc, leave=False, disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break

        try:
            adf, pvalue, *_ = sm.tsa.stattools.adfuller(view[i, :], regression="c", autolag="t-stat")
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute adf %s" % e, file=sys.stderr)
        out[i] = pvalue

    return out


def compute_emd(price: np.ndarray, window: int, out: np.ndarray, normalize: bool = True,
                cancellation_event: Optional[Event] = None) -> np.ndarray:
    view = sliding_window_view(price, window)
    assert out.shape[0] == view.shape[0]
    r = list(range(1000))
    random.shuffle(r)
    for i in tqdm(rand_range_8(len(view)), leave=False, desc="Computing emd", disable=trange_disabled):
        if cancellation_event and cancellation_event.is_set():
            break
        try:
            series = view[i, :]
            if normalize:
                series = (series - np.mean(series)) / np.std(series, ddof=1)
            emd = EMD(extrema_detection='parabol')
            IMFs = emd.emd(series, max_imf=3)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:
            print("unable to compute emd %s" % e, file=sys.stderr)
        else:
            sqrt_window = int(round(np.sqrt(window)))

            def compute_weights(y, sigma=None):
                """
                Computes weights for Hermite polynomial fitting using inverse-variance weighting.

                Args:
                    y (np.ndarray): Array of data points.
                    sigma (np.ndarray or float): Array of standard deviations (one per data point)
                                                 or a single value for constant standard deviation.

                Returns:
                    np.ndarray: Array of weights for each data point.
                """

                # Ensure sigma has the same shape as y for element-wise division
                if sigma is None:
                    sigma = np.std(y[:-sqrt_window], ddof=1)

                sigma = np.broadcast_to(sigma, y.shape)

                # Handle potential zero standard deviations (avoid division by zero)
                with np.errstate(invalid='ignore'):
                    weights = 1.0 / np.square(sigma)

                # Set weights to a small positive value (e.g., 1e-8) for zero standard deviations
                weights[sigma == 0] = 1e-8

                old_mean = np.mean(weights)

                weights[-sqrt_window:] /= np.arange(sqrt_window) + 1
                if old_mean > 0:
                    weights *= old_mean / np.mean(weights)

                return weights

            for j in range(min(3, len(IMFs) - 1)):
                imf = IMFs[j, :]
                out[i, j] = np.polynomial.hermite.hermfit(np.arange(len(imf)), imf, 3, w=compute_weights(imf))[0]


class IndexTracker:
    """
    A helper class to track and manage an index within a buffer.

    This class is used to efficiently manage the indexing of results within a buffer during calculations.
    It provides methods to track the current index, perform addition,
     and left shift operations for convenient access to specific slices of the buffer.

    Attributes:
        buffer (np.ndarray): The buffer array where the calculations are stored.
        index (int): The current index position within the buffer.

    """

    def __init__(self, buffer):
        """
        Initializes the IndexTracker object.

        Args:
            buffer (np.ndarray): The buffer array to be tracked.
        """

        self.buffer = buffer
        self.index = 0

    def __len__(self):
        """
        Returns the current index value.

        This method is equivalent to accessing the `index` attribute directly.

        Returns:
            int: The current index position within the buffer.
        """

        return self.index

    def __add__(self, other):
        """
        Performs addition on the current index and returns the previous value.

        This method allows for convenient calculation of offsets within the buffer.
        It adds the provided value to the current index, stores the previous index,
        and returns it.

        Args:
            other (int): The value to be added to the current index.

        Returns:
            int: The previous index value before the addition.
        """

        res = self.index
        self.index += other
        return res

    def __lshift__(self, other):
        """
        Performs left shift on the current index and returns the current value.

        This method allows for efficient calculation of offsets within the buffer.
        It adds the provided value to the current index and returns the new index.

        Args:
            other (int): The value to be added (left shifted) to the current index.

        Returns:
            int: The current index value after the left shift.
        """

        res = self.index
        self.index += other
        return self.index

    def __pos__(self):
        """
        Returns the current index value.

        This method allows for direct access to the current index using the unary plus (+) operator.

        Returns:
            int: The current index position within the buffer.
        """

        return self.index


def _process(path: Path, ohlcav: pd.DataFrame, max_threads=None, timeout=5 * 60):
    price = compute_price(ohlcav)
    log_price = compute_log_price(price)

    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        shuffled_window_choice = copy.copy(window_choice)
        random.shuffle(shuffled_window_choice)
        for window in shuffled_window_choice:
            cancellation_event = Event()
            buffer = _create_buffer(len(price), window)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The test statistic is outside of the range of p-values")
                warnings.filterwarnings("ignore", "invalid value encountered in sqrt")
                warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
                warnings.filterwarnings("ignore", "All-NaN slice encountered")

                g = IndexTracker(buffer)

                tasks = _submit_calculation(executor, cancellation_event, log_price, window, buffer, g)

                try:
                    concurrent.futures.wait(tasks, timeout=timeout)
                except concurrent.futures.TimeoutError:
                    pass
                finally:
                    cancellation_event.set()

                for task in tasks:
                    with contextlib.suppress(concurrent.futures.CancelledError):
                        if isinstance(e := task.exception(), KeyboardInterrupt):
                            raise e
                    if not task.done():
                        task.cancel()

                tasks.clear()

                tasks = _save_results(buffer=buffer, path=path, ohlcav=ohlcav, window=window, g=g, executor=executor,
                                      price=price)

                with contextlib.suppress(concurrent.futures.TimeoutError):
                    concurrent.futures.wait(tasks, timeout=1)

    return path


def _create_buffer(data_length: int, window: int) -> np.ndarray:
    res = np.empty((data_length - window + 1, 32), dtype=np.float32)
    res.fill(np.nan)
    return res


def _submit_calculation(executor: concurrent.futures.Executor, cancellation_event: Event,
                        log_price: np.ndarray, window: int, buffer: np.ndarray, g: IndexTracker):
    tasks = [
        executor.submit(compute_emd, log_price, window=window, out=buffer[:, +g:g << 3],
                        cancellation_event=cancellation_event),
        executor.submit(compute_hurst, log_price, window=window, out=buffer[:, g + 1], order=1),
        executor.submit(compute_dfa, log_price, window=window, out=buffer[:, g + 1], order=1,
                        cancellation_event=cancellation_event),
        executor.submit(compute_partial_auto_correlation, log_price, window=window,
                        out=buffer[:, +g:g << 3],
                        normalize=True, cancellation_event=cancellation_event),
        executor.submit(compute_auto_correlation, log_price, window=window, out=buffer[:, g + 1],
                        normalize=True, cancellation_event=cancellation_event),
        executor.submit(compute_mdfa, log_price, window=window, out=buffer[:, +g:g << (2 + 5)],
                        order=2 if window > 200 else 1,
                        cancellation_event=cancellation_event),
    ]
    # tasks.append(
    #    executor.submit(compute_matrix_profile_snippet, log_price, window=window, out=buffer[:, g + 1],
    #                    normalize=True))

    return tasks


def _save_results(buffer: np.ndarray, path: Path, ohlcav: pd.DataFrame, window: int, g: IndexTracker,
                  executor: concurrent.futures.Executor, price: np.ndarray) -> list[concurrent.futures.Future]:
    """
    Saves the calculated statistics & price to memory mapped files.

    Args:
        buffer: The buffer containing the calculated statistics.
        path: The path to the directory where the files will be saved.
        ohlcav: The original OHLC/Volume DataFrame.
        window: The window size used for the calculations.
        g: The IndexTracker object used for managing buffer indexing.
        executor: The thread pool executor used for asynchronous saving.
        price: The original price data.
    """

    min_time = int(ohlcav["time"].min() // 1e6)
    memmap = np.memmap(sanitize_filename(f'{path.stem.removesuffix(".csv")}_{min_time}_{window}_stats.npy'),
                       mode='w+',
                       dtype=np.float32,
                       shape=buffer[:, :+g].shape)

    def save(memmap, buffer):
        memmap[:] = buffer[:, :+g]
        memmap.flush()

    res = [executor.submit(save, memmap, buffer)]

    price_path = Path(sanitize_filename(f'{path.stem.removesuffix(".csv")}_{min_time}_price.npy'))
    if not price_path.exists():
        price_memmap = np.memmap(price_path, mode='w+', dtype=np.float64, shape=price.shape)

        def save(price_memmap, price):
            price_memmap[:] = price
            price_memmap.flush()

        res.append(executor.submit(save, price_memmap, price))

    return res


# pylint: disable=C901
def main():
    check_env()
    files = list_files()

    max_workers = os.getenv("MAX_WORKERS", None)
    if max_workers:
        max_workers = int(max_workers)
    if not max_workers:
        max_workers = os.cpu_count() or 1
    if max_workers < 0:
        max_workers = os.cpu_count() + max_workers

    print([x.name for x in files])

    max_runtime = os.getenv("MAX_RUNTIME", None)
    if max_runtime:
        max_runtime = int(max_runtime)
    if not max_runtime:
        max_runtime = 5 * 60 * 60

    start_time = time.monotonic()
    with tqdm() as pbar:
        while abs(time.monotonic() - start_time) < max_runtime:
            with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
                tasks: list[concurrent.futures.Future] = []
                for n in range(max_workers):
                    sub = None

                    while sub is None:
                        file = random.choice(files)
                        ohlcav = pd.read_csv(file, dtype={'time': 'int64',
                                                          'Open': 'float64',
                                                          'High': 'float64',
                                                          'Low': 'float64',
                                                          'Close': 'float64',
                                                          'Adj Close': 'float64',
                                                          'Volume': 'float64'})
                        ohlcav.rename(mapper=str.lower, axis=1, inplace=True)
                        sub = _pickup_sub_range(ohlcav)
                        if sub is None or len(sub) == 0:
                            print("unable to pick up sub range for %s", file.name, file=sys.stderr)
                            time.sleep(0.1)

                    pbar.set_description(f"processing file: {file.name}", refresh=True)
                    tasks.append(t := executor.submit(_process, file, sub))

                    def done_cb(t: concurrent.futures.Future):
                        if t.done():
                            pbar.update(1)
                            if (e := t.exception()) is None:
                                pbar.set_description(f"{t.result()}", refresh=True)
                            else:
                                print(e, file=sys.stderr)

                    t.add_done_callback(done_cb)

                try:
                    concurrent.futures.wait(tasks, timeout=60 * 60)
                except (concurrent.futures.TimeoutError, KeyboardInterrupt) as e:
                    print("stopping current batch", file=sys.stderr)
                    executor.shutdown(False, cancel_futures=True)
                    try:
                        import psutil
                    except ImportError:
                        pass
                    else:
                        for p in psutil.Process().children(recursive=True):
                            with contextlib.suppress(psutil.Error):
                                p.terminate()
                        with contextlib.suppress(concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
                            concurrent.futures.wait(tasks, timeout=5)
                        for p in psutil.Process().children(recursive=True):
                            with contextlib.suppress(psutil.Error):
                                p.kill()

                    if isinstance(e, KeyboardInterrupt):
                        raise

                for t in tasks:
                    if (e := t.exception()) is not None:
                        if isinstance(e, KeyboardInterrupt):
                            raise e
                tasks.clear()

                executor.shutdown(False, cancel_futures=True)


if __name__ == '__main__':
    main()
