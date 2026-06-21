# Agriculture Maproom — Setup & Run Instructions

Follow the steps below to set up and run the application.

## 1. Create the `data` folder

If it does not already exist, create a `data` folder in the same path where `app.py` is located.

## 2. Extract `shapes.zip`

Inside the `data` folder, extract the `shapes.zip` file.

## 3. Extract or add CSV files

- **First-time installation**: extract the `csv_files.zip` file inside the `data` folder.
- **Adding new data**: if you want to add new data afterward, place the new `.csv` files inside the `csv_files` folder (do not re-extract the zip).

## 4. Create the `assets` folder

If it does not already exist, create an `assets` folder inside the `agriculture` folder.

## 5. Configure the server (optional)

By default, the server runs on `localhost`, port `3333`. If you need to adjust these settings, edit the `config-dev-agriculture.yaml` file, located in the same path as `app.py`.

## 6. Run the application

From the path where `app.py` is located, run:

```bash
CONFIG=config-dev-agriculture.yaml pixi run --manifest-path agriculture/pixi.toml python app.py
```

## 7. View the maproom

Once the server is running, you can view the maproom at the address and port defined in `config-dev-agriculture.yaml`.

If you kept the default values, open the following link in your browser:

```
http://localhost:3333/python_maproom/agriculture/
```

