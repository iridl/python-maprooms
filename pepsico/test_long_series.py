# test_long_series.py

import xarray as xr
import numpy as np
import pandas as pd
import time
from memory_profiler import memory_usage


from app_calc import number_extreme_events_within_days
from app_calc import number_extreme_events_within_days_numpy
from app_calc import number_extreme_events_numpy_2

# ------------------------------------------------------------------
# Synthetic data generators
# ------------------------------------------------------------------

def make_daily_data_1d(ndays):
    """Generate 1D daily synthetic precipitation data"""
    time_coord = pd.date_range("1980-01-01", periods=ndays, freq="D")
    data = np.random.gamma(shape=2, scale=10, size=ndays)
    return xr.DataArray(
        data,
        dims="T",
        coords={"T": time_coord},
        name="pr"
    )


def make_daily_data_3d(ndays, nx, ny):
    """Generate 3D daily synthetic precipitation data"""
    time_coord = pd.date_range("1980-01-01", periods=ndays, freq="D")
    data = np.random.gamma(2, 10, size=(ndays, ny, nx))
    return xr.DataArray(
        data,
        dims=("T", "Y", "X"),
        coords={
            "T": time_coord,
            "Y": np.arange(ny),
            "X": np.arange(nx),
        },
        name="pr"
    )


# ------------------------------------------------------------------
# Function under stress test
# ------------------------------------------------------------------

def number_extreme_events_within_days(
    daily_data, threshold, window, dim="T"
):
    """Count extreme events within a maximum window of days"""
    count = 0
    dd = daily_data.copy()

    for w in range(1, window + 1):
        for t in range(len(dd[dim]) - (w - 1)):
            new_event = (
                dd.isel({dim: slice(t, t + w)})
                .sum(dim) > threshold
            )

            # mask days already used in an event
            dd[{dim: slice(t, t + w)}] = (
                dd[{dim: slice(t, t + w)}]
                .where(~new_event)
            )

            count = count + new_event

    return count


# ------------------------------------------------------------------
# Stress tests
# ------------------------------------------------------------------

def stress_test_time_1d(ndays, window, threshold, option=0):
    da = make_daily_data_1d(ndays)

    t0 = time.perf_counter()
    if option==0:
        out = number_extreme_events_within_days(
            da,
            threshold=threshold,
            window=window,
            dim="T"
            )
    elif option==1:
        out = number_extreme_events_within_days_numpy(
            da,
            threshold=threshold,
            window=window,
            dim="T"
        )
    elif option==2:
        out = number_extreme_events_numpy_2(
            da,
            threshold=threshold,
            window=window,
            dim="T"
        )
    elapsed = time.perf_counter() - t0

    print(
        f"[1D] n_serie_days={ndays:6d}, "
        f"window={window:2d}, "
        f"time={elapsed:6.2f}s, "
        f"output={out:2d}"
    )
    return out


def stress_test_time_3d(ndays, nx, ny, window, threshold, option=0):
    da = make_daily_data_3d(ndays, nx, ny)

    t0 = time.perf_counter()
    if option==0:
        out = number_extreme_events_within_days(
            da,
            threshold=threshold,
            window=window,
            dim="T"
            )
    elif option==1:
        out = number_extreme_events_within_days_numpy(
            da,
            threshold=threshold,
            window=window,
            dim="T"
        )
    elif option==2:
        out = number_extreme_events_numpy_2(
            da,
            threshold=threshold,
            window=window,
            dim="T"
        )
    elapsed = time.perf_counter() - t0

    print(
        f"[3D] n_serie_days={ndays:6d}, "
        f"grid={int(nx)}x{int(ny)}, "
        f"window={window:2d}, "
        f"time={elapsed:6.2f}s, "
        f"avg_output_per_grid_cell={int(out.mean().values):2d}"
 
    )
    return out

# ------------------------------------------------------------------
# Memory usage tests
# ------------------------------------------------------------------

def memory_test_1d(ndays, window, threshold, option=0):
    """Measure memory usage of number_extreme_events_within_days for 1D"""
    da = make_daily_data_1d(ndays)

    def run():
        if option==0:
            out = number_extreme_events_within_days(
                da,
                threshold=threshold,
                window=window,
                dim="T"
                )
        elif option==1:
            out = number_extreme_events_within_days_numpy(
                da,
                threshold=threshold,
                window=window,
                dim="T"
            )
        elif option==2:
            out = number_extreme_events_numpy_2(
                da,
                threshold=threshold,
                window=window,
                dim="T"
            )

    mem_usage = memory_usage(run)
    max_mem = max(mem_usage) - mem_usage[0]  # delta en MiB
    print(f"[MEM 1D] ndays={ndays}, \
          window={window}, \
          peak memory={max_mem:.1f} MiB"
          )
    return max_mem


def memory_test_3d(ndays, nx, ny, window, threshold,option=0):
    """Measure memory usage of number_extreme_events_within_days for 3D"""
    da = make_daily_data_3d(ndays, nx, ny)

    def run():
        if option==0:
            out = number_extreme_events_within_days(
                da,
                threshold=threshold,
                window=window,
                dim="T"
                )
        elif option==1:
            out = number_extreme_events_within_days_numpy(
                da,
                threshold=threshold,
                window=window,
                dim="T"
            )
        elif option==2:
            out = number_extreme_events_numpy_2(
                da,
                threshold=threshold,
                window=window,
                dim="T"
            )

    mem_usage = memory_usage(run)
    max_mem = max(mem_usage) - mem_usage[0]  # delta en MiB
    print(f"[MEM 3D] ndays={ndays}, \
          grid={ny}x{nx}, \
          window={window}, \
          peak memory={max_mem:.1f} MiB"
          )
    return max_mem

# ------------------------------------------------------------------
# Comparizon
# ------------------------------------------------------------------

def compare_1d(ndays, window, threshold):
    np.random.seed(0)
    da = make_daily_data_1d(ndays)

    out0 = number_extreme_events_within_days(
        da, threshold=threshold, window=window, dim="T"
    )
    # out1 = number_extreme_events_within_days_numpy(
    #     da, threshold=threshold, window=window, dim="T"
    # )
    out1 = number_extreme_events_numpy_2(
        da, threshold=threshold, window=window, dim="T"
    )

    print("option 0:", int(out0))
    print("option 1:", int(out1))
    print("IGUALES:", bool(out0 == out1))

def compare_3d(ndays, nx, ny, window, threshold):
    np.random.seed(0)
    da = make_daily_data_3d(ndays, nx, ny)

    out0 = number_extreme_events_within_days(
        da, threshold=threshold, window=window, dim="T"
    )
    # out1 = number_extreme_events_within_days_numpy(
    #     da, threshold=threshold, window=window, dim="T"
    # )
    out1 = number_extreme_events_numpy_2(
        da, threshold=threshold, window=window, dim="T"
    )

    print(out0.values)
    print("")
    print(out1.values)
    iguales = np.array_equal(out0.values, out1.values)
    #iguales = bool((out0 == out1).all())
    
    print("IGUALES:", iguales)

    if not iguales:
        #diff = out0 - out1
        diff = ( (out0 - out1) / out1 * 100).round(2)
        mask = diff != 0
        print("max diff:", diff.max().values)
        print("min diff:", diff.min().values)
        print(diff)

        num_points_gt0 = mask.sum().item() 
        total_points = diff.size
        percent_gt0 = (num_points_gt0 / total_points) * 100
        print(f"Puntos != 0: {num_points_gt0}")
        print(f"Porcentaje sobre el total: {percent_gt0:.2f}%")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    option=1
    n_days_window=8
    print("\n--- Stress test: long 1D time series ---")
    stress_test_time_1d(365, n_days_window, 80, option=option)
    stress_test_time_1d(3650, n_days_window, 80, option=option)     # 10 años
    stress_test_time_1d(10950, n_days_window, 80, option=option)    # 30 años
    #stress_test_time_1d(10950, 7, 80)
    #stress_test_time_1d(20000, 10, 80)
    print("\n--- Memory test: long 1D series ---")
    memory_test_1d(365, n_days_window, 80, option=option)
    memory_test_1d(3650, n_days_window, 80, option=option) 
    memory_test_1d(10950, n_days_window, 80, option=option)   # 30 años, 1D
    #memory_test_1d(20000, 10, 80)  # 55 años aprox, 1D

    print("\n--- Comparizon Function 0 vs Function 1 ---")
    compare_1d(10950, n_days_window, 80)

    print("\n--- Stress test: spatiotemporal (3D) ---")
    stress_test_time_3d(365, 60, 27, n_days_window, 80, option=option)
    stress_test_time_3d(3650, 60, 27, n_days_window, 80, option=option)
    stress_test_time_3d(10950, 60, 27, n_days_window, 80, option=option)    # 30 años


    print("\n--- Memory test: spatiotemporal 3D ---")
    memory_test_3d(365, 60, 27, n_days_window, 80, option=option) 
    memory_test_3d(3650, 60, 27, n_days_window, 80, option=option) 
    memory_test_3d(10950, 60, 27, n_days_window, 80,option=option)

    print("\n--- Comparizon Function 0 vs Function 1 ---")
    compare_3d(10950, 60, 27, n_days_window, 80)