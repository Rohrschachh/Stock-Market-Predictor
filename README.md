# Stock-Market-Predictor

Stocks Price Prediction app using LSTM and linear-Regression method of Machine leanring which run on a flask server which provides some API endpoint to get data related to any stocks listed on Yahoo finance and can also provided predicated closing price to stocks. The frontend application is made using Reactjs

## Getting Started

Python version above 3.3 is recommended.

<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `git clone https://github.com/Rohrschachh/Stock-Market-Predictor.git`.

<ins>**2. Configuring the dependencies:**</ins>

## Server

1. Run the [SetupWindows.bat](https://github.com/Rohrschachh/Stock-Market-Predictor/blob/master/Scripts/SetupWindows.bat) file found in `Scripts` folder. This will download the required packages and create a virtual environment for the project if they are not present yet.

2. One prerequisite is Python version above 3.3 should be installed.[Download](https://www.python.org/downloads)
3. After installation, run the [SetupWindows.bat](https://github.com/Rohrschachh/Stock-Market-Predictor/blob/master/Scripts/SetupWindows.bat) file again. If Python is installed properly, it will create a virtual environment and download the required packages. (This may take a longer amount of time)

4. After successfully install you can choose to run the server or Run the server manually after activating virtual environment and `cd server` and run command `python Server.py` in root folder of the server [http://localhost:5000](http://localhost:5000)

## Client

### Step-1 You will need

Please install them if you don't have them already.

- [node](https://nodejs.org/)
- [yarn](https://yarnpkg.com/en/docs/install) (Package manager)

### Step-2 Create a .env.local file

Create a .env.local file in client folder and paste

    VITE_SERVER_URL=http://localhost:5000

### Step-3 Install packages and run client

```shell
cd client
```

```shell
yarn && yarn dev
```

## License

Licensed under the Apache license.

---
