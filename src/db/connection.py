from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo="postgresql://golduser:goldpassword@localhost:5432/mydatabase"
)

def get_connection():
    return pool.connection()