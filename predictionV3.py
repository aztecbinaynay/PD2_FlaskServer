import requests
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import numba
import pickle
import sktime
import joblib
from datetime import datetime, date
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch, lfilter
from sktime.transformations.panel.rocket import Rocket
import statistics


def predictionv3(data):
    def ODI_finder(desat_min, Time, SpO2_list):
        ODI = 0
        window_begin = 0
        window_end = 120
        iter = 10
        while True:
            try:
                index_begin = Time.index(window_begin)
                print(window_begin, index_begin)
                index_end = Time.index(window_end)
                print(window_end, index_end)
                SpO2_mean = round(statistics.mean(
                    SpO2_list[index_begin:index_end]))
                print(SpO2_mean)
                index10 = Time.index(window_end+10)
                print(window_end+10, index10)
                SpO2_list_10secs = SpO2_list[index_end:index10]
                print("len of list is:", len(SpO2_list_10secs))
                print("unique numbers are: ", list(set(SpO2_list_10secs)))
                conditions_met = True
                for i in SpO2_list_10secs:
                    if SpO2_mean - i >= desat_min:
                        continue
                    else:
                        conditions_met = False
                        break
                if conditions_met:
                    ODI += 1
                window_begin += 10
                window_end += 10
                print("\n")

            except Exception as error:
                print("error is:", error)
                print("ODI is:", ODI)
                break
        print("I am here returning home!")
        return ODI

    def round_down_and_cap(series):
        series = series.round(0)
        series[series < 0] = 0
        series[series > 100] = 100
        series = series.astype(np.int64)
        series[series == -0.0] = 0
        series = np.array(series)
        return series

    def Apply_Notch_Butterworth_v2(x):
        # Butterworth Band-pass filter parameters
        fs = 200  # Sampling frequency in Hz
        lowcut = 0.3  # ! recoommended by AASM Lower cutoff frequency in Hz
        highcut = 70  # ! recoommended by AASM Upper cutoff frequency in Hz
        order = 4  # Filter order

        # Design the Butterworth filter
        b, a = butter(
            order,
            [lowcut, highcut],
            btype="band",
            fs=fs,
        )

        # Notch filter parameters
        notch_freq = 50  # ! Notch frequency in Hz toremove powerline interference
        Q = 30  # Quality factor
        # Design the Notch filter
        b_notch, a_notch = iirnotch(notch_freq, Q, fs=fs)

        #! previous order of filter application was Butterworth then Notch

        # Apply the Notch filter
        filtered_signal = filtfilt(b_notch, a_notch, x.values)

        # Apply the Butterworth filter
        filtered_signal = filtfilt(b, a, filtered_signal)

        return filtered_signal

    def Apply_Bandpass_filter(x):
        # Butterworth Band-pass filter parameters
        fs = 200  # Sampling frequency in Hz
        lowcut = 10  # ! recoommended by AASM Lower cutoff frequency in Hz
        highcut = 70  # ! recoommended by AASM Upper cutoff frequency in Hz
        order = 4  # Filter order

        # Design the Butterworth filter
        b, a = butter(
            order,
            [lowcut, highcut],
            btype="band",
            fs=fs,
        )

        # Apply the Butterworth filter
        filtered_signal = filtfilt(b, a, x.values)

        return filtered_signal

    def cubic_spline_interpolation(data):
        # Define the original 300 data point signal
        l = len(data)

        x = np.linspace(0, 30, l)
        y = data.values

        # Normalize the data
        scaler = StandardScaler()
        y_norm = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Define the new time points for upsampling
        x_new = np.linspace(0, 30, 6000)

        # Upsample using cubic spline interpolation
        f_cubic = interp1d(x, y_norm, kind="cubic")
        y_cubic = f_cubic(x_new)

        # Denormalize the data
        y_rescaled = scaler.inverse_transform(y_cubic.reshape(-1, 1)).flatten()

        # Clip the values to the original data range
        y_rescaled = np.clip(y_rescaled, y.min(), y.max())

        y_rescaled_series = pd.Series(y_rescaled)
        return y_rescaled_series

    AHI_table = {
        "Severity": None,
        "AHI": None,
        "TimeIn": data["TimeIn"],
        "TimeOut": data["TimeOut"],
        "UserID": data["UserID"],
        "Normal": 0,
        "Apnea": 0,
        "Hypopnea": 0,
        "MT": 0,
        "avg_HR": 0,
        "lowest_HR": 0,
        "highest_HR": 0,
        "ODI3": 0,
        "ODI4": 0,
        "lowest_SpO2": 0,
        "avg_SpO2": 0,
        "highest_SpO2": 0,
        "repeat_study": "",
        "recommendations": ""
    }
    print("I am here")
    print(AHI_table)
    print(type(data["ECG"]))
    try:
        for i in data.keys():
            if i not in ["UserID", "TimeIn", "TimeOut"]:
                data[i] = ast.literal_eval(data[i])
        print("turned the strings into lists")
    except Exception as e:
        print(e)
    excluded_columns = [
        "UserID",
        "TimeOut",
        "TimeIn",
    ]  # Columns to exclude from the copy
    new_dict = {
        key: value for key, value in data.items() if key not in excluded_columns
    }
    print("created a new dict")
    import pandas as pd
    df = pd.DataFrame(new_dict, columns=new_dict.keys())
    print("line 187")
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%H:%M:%S").dt.time
    print("line 190")
    df["Timestamp"] = df["Timestamp"].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    print("line 193")
    # Handle the rollover from 24:36:00 to 0:0:06:42
    try:
        rollover_index = df[df["Timestamp"] < df["Timestamp"].shift()].index[0]
        df.loc[rollover_index:, "Timestamp"] += 24 * 3600
    except Exception as e:
        pass
    print("line 198")
    print("created the dataframe")
    print(df.head(3))
    print(df.tail(3))
    print(len(df))
    # Add MT in hours to the AHI table
    MT_seconds = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    MT_hrs = round(MT_seconds / 3600, 2)
    AHI_table["MT"] = MT_hrs
    if AHI_table["MT"] < 2:
        AHI_table["repeat_study"] = "Insufficient Sleep Duration. Repeat sleep study"
    else:
        AHI_table["repeat_study"] = "Sufficient Sleep Duration. Repeat of study subject to medical professional's discretion."
    # add the lowest, highest, and average heart rate to the AHI table
    AHI_table["avg_HR"] = round(df["PulseRate"].mean(), 2)
    AHI_table["lowest_HR"] = round(df["PulseRate"].min(), 2)
    AHI_table["highest_HR"] = round(df["PulseRate"].max(), 2)
    # add the lowest, highest, and average SpO2to the AHI table
    AHI_table["avg_SpO2"] = round(df["SpO2"].mean())
    AHI_table["lowest_SpO2"] = round(df["SpO2"].min())
    AHI_table["highest_SpO2"] = round(df["SpO2"].max())
    print(AHI_table["avg_SpO2"], AHI_table["lowest_SpO2"],
          AHI_table["highest_SpO2"])
    df["TimestampDiff"] = df["Timestamp"] - df["Timestamp"].shift(1)
    df["TimestampDiff"] = df["TimestampDiff"].fillna(0)
    df["TimestampCumSum"] = df["TimestampDiff"].cumsum()
    Time = df["TimestampCumSum"].tolist()
    SpO2_list = df["SpO2"].tolist()
    ODI3 = ODI_finder(3, Time, SpO2_list)
    print(ODI3)
    ODI4 = ODI_finder(4, Time, SpO2_list)
    print(ODI4)
    print(ODI3, ODI4)
    AHI_table["ODI3"] = ODI3
    AHI_table["ODI4"] = ODI4
    print(AHI_table["ODI3"], AHI_table["ODI4"])
    # Filter the DataFrame to get data from the first 5 minutes
    print("created the dataframe")
    print(df.head(3))
    print(df.tail(3))
    print(len(df))
    # Filter the DataFrame to get data from the first 5 minutes
    new_df = df.copy()
    l_ecg = []
    l_airflow = []
    l_snore = []
    l_temp = []
    l_spo2 = []
    l_pulserate = []
    try:
        while True:
            # Convert the first timestamp to a datetime object with today's date
            first_timestamp = new_df["Timestamp"][0]

            # Add 30 seconds
            end_time = first_timestamp + 30

            # Select the rows
            thirty_sec = new_df[new_df["Timestamp"] <= end_time]
            l_ecg.append(thirty_sec["ECG"].values)
            l_airflow.append(thirty_sec["AirFlow"].values)
            l_snore.append(thirty_sec["Snore"].values)
            l_temp.append(thirty_sec["Temp"].values)
            l_spo2.append(thirty_sec["SpO2"].values)
            l_pulserate.append(thirty_sec["PulseRate"].values)
            new_df = new_df.drop(range(len(thirty_sec)))
            new_df = new_df.reset_index(drop=True)
    except Exception as e:
        print(e)

    # Example column names
    column_names = ["ECG", "AirFlow", "Snore", "Temp", "SpO2", "PulseRate"]

    # Creating a DataFrame
    data_now = {
        "ECG": l_ecg,
        "AirFlow": l_airflow,
        "Temp": l_temp,
        "Snore": l_snore,
        "SpO2": l_spo2,
        "PulseRate": l_pulserate,
    }
    print("created dataframe from data_now")
    print(
        data_now["ECG"],
        len(data_now["AirFlow"]),
        len(data_now["Temp"]),
        len(data_now["Snore"]),
        len(data_now["SpO2"]),
        len(data_now["PulseRate"]),
    )
    try:
        df_v2 = pd.DataFrame(data_now, columns=column_names)
    except Exception as e:
        print(e)
    df_v2_series = df_v2.applymap(lambda x: pd.Series(x.tolist()))
    # create a new empty dataframe with the same shape as the original dataframe
    new_df = pd.DataFrame(index=df_v2_series.index,
                          columns=df_v2_series.columns)

    # iterate over each cell in the dataframe
    for i in range(df_v2_series.shape[0]):
        for j in range(df_v2_series.shape[1]):
            cell_value = df_v2_series.iloc[i, j]
            # check if the cell needs to be upsampled/downsampled
            if isinstance(cell_value, pd.Series) and len(cell_value) != 6000:
                # apply the cubic_spline_interpolation function to the cell value
                new_series = cubic_spline_interpolation(cell_value)
                # fill the new dataframe with the upsampled/downsampled values
                new_df.iloc[i, j] = new_series
            else:
                # if the cell doesn't need to be upsampled/downsampled, fill the new dataframe with the original value
                new_df.iloc[i, j] = cell_value
    print("finished upsampling")
    # replace the old dataframe with the new one
    df_v2_series_upsampled = new_df
    df_v2_series_upsampled = df_v2_series_upsampled.applymap(
        lambda x: pd.Series(x.tolist())
    )

    df_v2_series_upsampled["ECG_filtered"] = df_v2_series_upsampled["ECG"].apply(
        Apply_Notch_Butterworth_v2
    )
    print("finsihed ECG filters")

    df_v2_series_upsampled["Snore_filtered"] = df_v2_series_upsampled["Snore"].apply(
        Apply_Bandpass_filter
    )
    print("finsihed Snore filters")

    df_v2_series_upsampled["SpO2_v2"] = df_v2_series_upsampled["SpO2"].apply(
        round_down_and_cap
    )
    print("finsihed SpO2 rounding")

    df_v2_series_upsampled = df_v2_series_upsampled.applymap(
        lambda x: pd.Series(x) if isinstance(
            x, list) else pd.Series(x.tolist())
    )
    print("finsihed turning arrays into pandas")

    df_final = df_v2_series_upsampled[
        ["ECG_filtered", "Snore_filtered", "AirFlow", "Temp", "SpO2_v2", "PulseRate"]
    ]

    rocket_transform = joblib.load(
        r"C:\Users\DeadSpheres\Desktop\pd_2\RocketTransV3.pkl"
    )
    print("imported rocket transform")
    print(df_final.head(5))
    print(df_final.tail(5))
    X_values = rocket_transform.transform(df_final)
    X_train_df = pd.DataFrame(X_values)
    model = joblib.load(
        r"C:\Users\DeadSpheres\Desktop\pd_2\new_models\rf_model_v3.pkl"
    )
    print("imported the model")
    y_pred = model.predict(X_train_df)

    def create_integer_counts_dict(arr, AHI_table):
        for num in y_pred:
            if num == 0:
                AHI_table["Normal"] += 1
            elif num == 1:
                AHI_table["Apnea"] += 1
            elif num == 2:
                AHI_table["Hypopnea"] += 1

        # Convert the datetime strings into datetime objects
        datetime_str1 = AHI_table["TimeIn"]
        datetime_str2 = AHI_table["TimeOut"]
        datetime_format = "%Y-%m-%d %H:%M:%S"  # Format of the datetime strings

        datetime_obj1 = datetime.strptime(datetime_str1, datetime_format)
        datetime_obj2 = datetime.strptime(datetime_str2, datetime_format)

        # Calculate the time difference in hours
        time_difference = (
            datetime_obj2 - datetime_obj1).total_seconds() / 3600
        print("time diff:", time_difference)

        # Calculate the AHI
        AHI = (AHI_table["Apnea"] + AHI_table["Hypopnea"]) / time_difference

        # Add the AHI to the dictionary
        AHI_table["AHI"] = AHI
        print("AHI: ", AHI)

        # Add the severity to the dictionary

        if AHI < 5:
            AHI_table["Severity"] = "Normal"
            AHI_table["recommendations"] = "Normal sleep pattern. Confirm with medical professional for further analysis."
        elif AHI >= 5 and AHI < 15:
            AHI_table["Severity"] = "Mild"
            AHI_table["recommendations"] = "Mild sleep apnea. Confirm with medical professional for further analysis. Primary treatments include positional therapy, oral appliance, or surgery - UPPP. Secondary treatments include CPAP, BiPAP, or APAP. Adjunctive therapy includes weight loss, exercise, and smoking cessation. Consult with your medical provider for the best treatment option for you."
        elif AHI >= 15 and AHI < 30:
            AHI_table["Severity"] = "Moderate"
            AHI_table["recommendations"] = "Moderate sleep apnea. Confirm with medical professional for further analysis. Primary treatments include CPAP, BiPAP, or APAP. Secondary treatments include Oral appliance or surgery - UPPP. Adjunctive therapy includes weight loss, positional therapy, exercise, and smoking cessation. Consult with your medical provider for the best treatment option for you."
        elif AHI >= 30:
            AHI_table["Severity"] = "Severe"
            AHI_table["recommendations"] = "Severe sleep apnea. Confirm with medical professional for further analysis. Primary treatments include CPAP, BiPAP, or APAP. Secondary treatments include Oral appliance or surgery - MMA. Adjunctive therapy includes weight loss, positional therapy, exercise, and smoking cessation. Consult with your medical provider for the best treatment option for you."

        return AHI_table

    print("creating the final AHI table")
    AHI_table = create_integer_counts_dict(y_pred, AHI_table)
    print(AHI_table)
    return AHI_table
