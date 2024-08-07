import concurrent.futures
import contextlib
import copy
import os
import random
import re
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


def random_chunks_of_eight(length: int) -> np.ndarray:
    """
    Randomly shuffles a sequence of numbers into chunks of 8.

    The function creates a sequence of numbers from 0 to length - 1,
    reshapes it into chunks of 8 (if possible), shuffles the chunks,
    and finally flattens the result. If the length is not divisible by 8,
    the remaining elements are shuffled and inserted randomly into the
    chunks.

    Args:
        length: The length of the sequence to be shuffled.

    Returns:
        A numpy array of length `length` with the shuffled numbers.
    """

    res = np.arange(length)

    if length % 8 == 0:
        res = res.reshape(-1, 8)
        np.random.shuffle(res)
        return res.flatten()

    # Handle case where length is not divisible by 8
    num_full_chunks = length // 8
    full_chunks = res[:num_full_chunks * 8].reshape(-1, 8)
    _random_generator.shuffle(full_chunks)
    remaining = res[num_full_chunks * 8:]
    _random_generator.shuffle(remaining)

    # Insert remaining elements randomly into chunks
    insert_idx = np.random.randint(num_full_chunks + 1)
    full_chunks = np.insert(full_chunks, insert_idx, remaining, axis=0)
    return full_chunks.flatten()


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
    for i in tqdm(random_chunks_of_eight(len(view)), desc=desc, leave=False, disable=trange_disabled):
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

    for i in tqdm(random_chunks_of_eight(len(view)), leave=False, desc="Computing mdfa", disable=trange_disabled):
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

    for i in tqdm(random_chunks_of_eight(len(view)), leave=False, desc="Computing dfa", disable=trange_disabled):
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
    for i in tqdm(random_chunks_of_eight(len(view)), leave=False, desc="Computing emd", disable=trange_disabled):
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


def _process(path: Path, ohlcav: pd.DataFrame, max_threads=None, timeout=5 * 60, per_window_timeout=5 * 60):
    price = compute_price(ohlcav)
    log_price = compute_log_price(price)
    start_time = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        shuffled_window_choice = copy.copy(window_choice)
        random.shuffle(shuffled_window_choice)
        for window in shuffled_window_choice:
            if time.monotonic() - start_time > timeout:
                break
            cancellation_event = Event()
            buffer = _create_buffer(len(price), window)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The test statistic is outside of the range of p-values")
                warnings.filterwarnings("ignore", "invalid value encountered in sqrt")
                warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
                warnings.filterwarnings("ignore", "All-NaN slice encountered")

                g = IndexTracker(buffer)

                tasks = _submit_calculation(executor, cancellation_event, log_price, window, buffer, g)

                if not tasks:
                    continue

                task_timeout = min(per_window_timeout, timeout - int(time.monotonic() - start_time))
                with contextlib.suppress(concurrent.futures.TimeoutError):
                    concurrent.futures.wait(tasks, timeout=task_timeout)

                cancellation_event.set()

                with contextlib.suppress(concurrent.futures.TimeoutError):
                    concurrent.futures.wait(tasks, timeout=min(30, task_timeout))

                for task in tasks:
                    if not task.done():
                        task.cancel()
                        continue
                    with contextlib.suppress(concurrent.futures.CancelledError):
                        if isinstance(e := task.exception(), KeyboardInterrupt):
                            raise e

                tasks.clear()

                tasks = _save_results(buffer=buffer, path=path, ohlcav=ohlcav, window=window, g=g, executor=executor,
                                      price=price)

                with contextlib.suppress(concurrent.futures.TimeoutError):
                    concurrent.futures.wait(tasks, timeout=30)

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


def parse_time_interval(time_str: str | int) -> int:
    """Parses a time interval string into seconds.

    Args:
        time_str: The time interval string to parse.

    Returns:
        The equivalent number of seconds.

    Raises:
        ValueError: If the time string is invalid.
    """

    with contextlib.suppress(ValueError):
        return int(time_str)
    time_units = {"s": 1, "m": 60, "h": 3600}
    match = re.findall(r"(\d+(?:\.\d+)?)([smh])", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    total_seconds = sum(float(value) * time_units[unit] for value, unit in match)
    return int(total_seconds)


def configure_environment():
    """Configures environment variables and returns relevant parameters."""
    max_workers = float(os.getenv("MAX_WORKERS", os.cpu_count() or 1))
    if max_workers < 0:
        max_workers = os.cpu_count() + max_workers
    if 0.0 < max_workers < 1.0:
        max_workers = int(os.cpu_count() * max_workers)

    default_max_runtime = f'{5 * 60 * 60}'
    max_runtime = parse_time_interval(os.getenv("MAX_RUNTIME", default_max_runtime) or default_max_runtime)

    return int(max_workers), max_runtime


def pick_and_process_file(files: list[Path], executor: concurrent.futures.Executor, pbar, timeout: float):
    start_time = time.monotonic()

    """Processes a single file using the provided executor."""
    while True:
        file = random.choice(files)
        ohlcav = pd.read_csv(
            file,
            dtype={
                "time": "int64",
                "Open": "float64",
                "High": "float64",
                "Low": "float64",
                "Close": "float64",
                "Adj Close": "float64",
                "Volume": "float64",
            },
        )
        ohlcav.rename(mapper=str.lower, axis=1, inplace=True)
        sub = _pickup_sub_range(ohlcav)
        if sub is not None and len(sub) > 0:
            break
        print(f"Failed to pick sub-range for {file.name}, retrying...", file=sys.stderr)
        time.sleep(0.1)
        if abs(time.monotonic() - start_time) > timeout:
            raise TimeoutError(f"Failed to pick sub-range for {file.name} within {timeout} seconds")

    timeout = max(timeout, 60)
    task = executor.submit(_process, file, sub, timeout=timeout)

    def done_callback(future):
        if future.done():
            exception = future.exception()
            if exception is None:
                result = future.result()
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description(f"Processed: {result}", refresh=True)
            else:
                print(f"Error: {exception}", file=sys.stderr)

    task.add_done_callback(done_callback)
    return file, task


def handle_timeout(executor, tasks: dict[Path, concurrent.futures.Future], timeout: float = 60 * 60):
    """Handles timeouts and process termination."""
    try:
        concurrent.futures.wait(tasks.values(), timeout=timeout)
    except (concurrent.futures.TimeoutError, KeyboardInterrupt) as e:
        print(f"Stopping current batch due to {e}", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)

        # Attempt forceful termination
        with contextlib.suppress(ImportError):
            import psutil

            for child in psutil.Process().children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    child.terminate()

            with contextlib.suppress(concurrent.futures.TimeoutError):
                concurrent.futures.wait(tasks.values(), timeout=5)

            for child in psutil.Process().children(recursive=True):
                with contextlib.suppress(psutil.Error):
                    child.terminate()

        if isinstance(e, KeyboardInterrupt):
            raise


def main():
    max_workers, max_runtime = configure_environment()
    check_env()
    files = list_files()

    count = 0
    with tqdm() as pbar:
        start_time = time.monotonic()
        while abs(time.monotonic() - start_time) < max_runtime:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                timeout = max(int(max_runtime - (time.monotonic() - start_time)), 10)
                timeout = min(timeout, 60 * 60)

                tasks: dict[Path, concurrent.futures.Future]
                tasks = dict(pick_and_process_file(files, executor, pbar, timeout) for _ in range(max_workers))
                handle_timeout(executor, tasks, timeout=timeout)

                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                        continue
                    exception = task.exception()
                    if isinstance(exception, KeyboardInterrupt):
                        executor.shutdown(False, cancel_futures=True)
                        raise exception

                    count += 1

    print(f"Finished processing {count} files.")


if __name__ == '__main__':
    main()
