import xarray as xr
from pathlib import Path


TOP_PATH = Path("/Data/data24")
COMMON_PATH = f'ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global'
INPUT_PATH = TOP_PATH / COMMON_PATH / "daily"
OUTPUT_PATH = TOP_PATH / COMMON_PATH / "monthly"

for scenario_path in INPUT_PATH.iterdir():
    for model_path in scenario_path.iterdir():
        for var in [
            #"hurs", "huss", "pr", "prsn", "ps", "rlds", "sfcwind", "tas", "tasmax",
            #"tasmin",
            "rsds",
        ]:
            print(model_path / "zarr" / var)
            daily = xr.open_zarr(model_path / "zarr" / var)
            monthly = daily.resample(T="MS").mean().chunk({"T": 120})
            output_path = (
                OUTPUT_PATH / scenario_path.stem / model_path.stem / "zarr" / var
            )
            print(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            monthly.to_zarr(store=output_path)
