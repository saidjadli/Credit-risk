import psycopg

def get_connection():
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="golduser",
        user="goldpassword",
        password="mypassword"
    )