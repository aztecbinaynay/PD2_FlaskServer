from flask import Flask, request, jsonify
import sqlite3
from predictionV3 import predictionv3
import concurrent.futures
from datetime import datetime


app = Flask(__name__)

executor = concurrent.futures.ThreadPoolExecutor()


@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/insert", methods=["POST"])
def insert_data():
    try:
        data = request.get_json()  # Get the JSON data from the request
        conn = sqlite3.connect(
            r"C:\Users\DeadSpheres\Downloads\WebApp-FinalDefense\flask-vue-edosa\resultsbackend\SensorReadings.db")
        cursor = conn.cursor()

        # Begin a transaction
        conn.execute("BEGIN TRANSACTION")

        cursor.execute(
            """
            INSERT INTO SensorReadings (UserID, Therm, ECG, Airflow, Snore, SpO2, HR, TimeIn, TimeOut, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                data["UserID"],
                data["Temp"],
                data["ECG"],
                data["AirFlow"],
                data["Snore"],
                data["SpO2"],
                data["PulseRate"],
                data["TimeIn"],
                data["TimeOut"],
                data["Timestamp"],
            ),
        )

        # Commit the transaction if all insertions are successful
        conn.commit()
        conn.close()
        executor.submit(prediction_task, data)
        print("Data inserted")
        return "Data inserted"
    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        conn.close()
        print("Error", e)
        # Log the caught exception
        app.logger.error(f"Error inserting data: {str(e)}")
        return "Data not inserted due to an error", 500


def prediction_task(data):
    try:
        print("prediction starting")
        start = datetime.now()
        result = predictionv3(data)
        end = datetime.now()
        duration = end - start
        print("inference time completed in",
              duration.total_seconds(), "seconds")
        print("prediction end")
        print("Data inserting...")
        conn2 = sqlite3.connect(
            r"C:\Users\DeadSpheres\Downloads\WebApp-FinalDefense\flask-vue-edosa\resultsbackend\SensorReadings.db")
        cursor2 = conn2.cursor()

        # Begin a transaction
        conn2.execute("BEGIN TRANSACTION")

        cursor2.execute(
            """
            INSERT INTO AHI_table (Severity, AHI, TimeIn, TimeOut, UserID, Normal, Apnea, Hypopnea, MT, avg_HR, lowest_HR, highest_HR, ODI3, ODI4, lowest_SpO2, avg_SpO2, highest_SpO2, repeat_study, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result["Severity"],
                result["AHI"],
                result["TimeIn"],
                result["TimeOut"],
                result["UserID"],
                result["Normal"],
                result["Apnea"],
                result["Hypopnea"],
                result["MT"],
                result["avg_HR"],
                result["lowest_HR"],
                result["highest_HR"],
                result["ODI3"],
                result["ODI4"],
                result["lowest_SpO2"],
                result["avg_SpO2"],
                result["highest_SpO2"],
                result["repeat_study"],
                result["recommendations"],
            ),
        )

        # Commit the transaction if all insertions are successful
        conn2.commit()
        conn2.close()
        print("Data inserted in AHI table")
    except Exception as e:
        # Rollback the transaction if an error occurs
        conn2.rollback()
        conn2.close()
        print("Error", e)


@app.route("/retrieveUserData", methods=["POST"])
def getUser_data():
    try:
        data = request.get_json()  # Get the JSON data from the request
        conn = sqlite3.connect(
            r"C:\Users\DeadSpheres\Desktop\pd_2\SensorReadings.db")
        cursor = conn.cursor()

        # Begin a transaction
        conn.execute("BEGIN TRANSACTION")

        cursor.execute(
            "SELECT UserID, TimeIn, TimeOut FROM SensorReadings WHERE UserID=?",
            (data["UserID"],),
        )

        # Fetch all the rows from the result
        rows = cursor.fetchall()

        if rows:
            # Create a list to store the retrieved data
            data_list = []
            for row in rows:
                UserID, TimeIn, TimeOut = row
                time_dict = {
                    "TimeIn": TimeIn,
                    "TimeOut": TimeOut,
                }
                data_list.append(time_dict)

            # Create a dictionary to store the final response
            response_dict = {
                "UserID": data["UserID"],
                "Time": data_list,
            }

            conn.commit()
            conn.close()
            print("Data retrieved")
            return jsonify(response_dict), 200

        conn.commit()
        conn.close()
        print("Data unavailable")
        return "No data found for the given UserID", 404

    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        conn.close()
        print("Error", e)
        return str(e), 500


@app.route("/retrieveUserInstance", methods=["POST"])
def getInstance_data():
    try:
        data = request.get_json()  # Get the JSON data from the request
        conn = sqlite3.connect(
            r"C:\Users\DeadSpheres\Desktop\pd_2\SensorReadings.db")
        cursor = conn.cursor()

        # Begin a transaction
        conn.execute("BEGIN TRANSACTION")

        cursor.execute(
            "SELECT * FROM SensorReadings WHERE UserID=? AND TimeIn=? AND TimeOut=?",
            (data["UserID"], data["TimeIn"], data["TimeOut"]),
        )

        # Fetch the row from the result
        row = cursor.fetchone()

        if row is not None:
            # Extract the values from the row
            (
                id_,
                UserID,
                Therm,
                ECG,
                Airflow,
                Snore,
                SpO2,
                HR,
                TimeIn,
                TimeOut,
                timestamp,
            ) = row

            # Create a dictionary to store the retrieved data
            data_dict2 = {
                "UserID": UserID,
                "Therm": Therm,
                "ECG": ECG,
                "Airflow": Airflow,
                "Snore": Snore,
                "SpO2": SpO2,
                "HR": HR,
                "TimeIn": TimeIn,
                "TimeOut": TimeOut,
                "Timestamp": timestamp,
            }

            # Retrieve data from AHI_table
            cursor.execute(
                "SELECT Severity, AHI, Normal, Apnea, Hypopnea FROM AHI_table WHERE UserID=? AND TimeIn=? AND TimeOut=?",
                (data["UserID"], data["TimeIn"], data["TimeOut"]),
            )

            # Fetch the row from the result
            ahi_row = cursor.fetchone()

            if ahi_row is not None:
                # Extract the values from the AHI row
                Severity, AHI, Normal, Apnea, Hypopnea = ahi_row

                # Create a dictionary to store the retrieved data from AHI_table
                ahi_dict = {
                    "Severity": Severity,
                    "AHI": AHI,
                    "Normal": Normal,
                    "Apnea": Apnea,
                    "Hypopnea": Hypopnea,
                }

                # Combine the data dictionaries
                data_dict2.update(ahi_dict)

            conn.commit()
            conn.close()
            print("Data retrieved")
            return jsonify(data_dict2), 200

        conn.commit()
        conn.close()
        print("Data unavailable")
        return "row does not exist", 404

    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.rollback()
        conn.close()
        print("Error", e)
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

