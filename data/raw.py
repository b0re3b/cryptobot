import psycopg2
from utils import USERPSG, PASSWD,DATABASE

conn = psycopg2.connect("dbname=crypto")