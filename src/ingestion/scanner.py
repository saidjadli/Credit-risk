import os

def scan_incoming_folder(path):
    files = os.listdir(path)
    return [f for f in files if f.endswith(".csv")]