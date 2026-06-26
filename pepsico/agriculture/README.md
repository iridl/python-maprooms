# Agriculture Maproom — Setup & Run Instructions

Follow the steps below to set up and run the application.

## 1. Create the `data` folder

If it does not already exist, create a `data` folder in the same path where `app.py` is located.

## 2. Extract `shapes.zip`

Inside the `data` folder, extract the `shapes.zip` file.

## 3. Extract or add CSV files

- **First-time installation**: extract the `csv_files.zip` file inside the `data` folder.
- **Adding new data**: if you want to add new data afterward, place the new `.csv` files inside the `csv_files` folder (do not re-extract the zip).

## 4. Configure Users

Rename the `users.txt` file to `.users`.

By default, the application grants access with the following credentials:

- **Username:** `pepe`
- **Password:** `hola`

To change the password or add additional users, see the **Users** section below.

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
# Maproom Users

To add or update a user, follow these steps.

From the directory where `app.py` is located, run:

```bash
CONFIG=config-dev-agriculture.yaml pixi run --manifest-path agriculture/pixi.toml python
```

Then execute:

```python
from agriculture import extrafunctions as ext

ext.hash_user_password('**user**', '**password**')
```

Replace **user** and **password** with the desired username and password.

For example:

```python
ext.hash_user_password('pepe', 'pepe123')
```

This will return a value similar to:

```text
pepe:$2b$12$fYfe4bfomqHLqOLLm53SiuNp2Z9rEKMcvatc.DXMeDt/uGRaegOs.
```

Copy the returned value (without the surrounding single quotes, if present) and paste it into your `.users` file.

If the user already exists, this operation is considered a password update. Simply replace the existing line corresponding to that user in the `.users` file with the newly generated value.

> **⚠️ Warning:** Ensure that the `admin` user is always present in the `.users` file, as it is required for the application to function correctly.
