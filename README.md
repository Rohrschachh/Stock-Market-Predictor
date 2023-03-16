# Stock-Market-Predictor

stocks.py is the main python file.

stocks_data folder will be created on project startup which contains the data files from where the results are to be calibrated
It also includes the data of specified Ticker Symbols and their Dates.

## Getting Started

Python version above 3.3 is recommended.

<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `git clone https://github.com/Rohrschachh/Stock-Market-Predictor.git`.

<ins>**2. Configuring the dependencies:**</ins>

For Windows

1. Run the [SetupWindows.bat](https://github.com/Rohrschachh/Stock-Market-Predictor/blob/master/Scripts/SetupWindows.bat) file found in `Scripts` folder. This will download the required packages and create a virtual environment for the project if they are not present yet.

2. One prerequisite is Python version above 3.3 should be installed.[Download](https://www.python.org/downloads)
3. After installation, run the [SetupWindows.bat](https://github.com/Rohrschachh/Stock-Market-Predictor/blob/master/Scripts/SetupWindows.bat) file again. If Python is installed properly, it will create a virtual environment and download the required packages. (This may take a longer amount of time)

4. After successfully install you can choose to run the app or Run the app manually after activating virtual environment use command `streamlit run stocks.py` in root folder which will start the Streamlit App on [http://localhost:8501](http://localhost:8501)
