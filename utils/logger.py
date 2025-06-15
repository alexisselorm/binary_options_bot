import csv
from datetime import datetime


def log_trade(data, filename='trade_log.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
