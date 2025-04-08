import psycopg2
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

conn = psycopg2.connect('postgresql://postgres:28488476@localhost:5433/crypto')
cur = conn.cursor()
cur.execute ("SELECT * FROM binance")
rows = cur.fetchall()
print(rows)