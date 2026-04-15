import shutil
import os
def archive_files(files, src, dest):
    for f in files:
        shutil.move(os.path.join(src, f), os.path.join(dest, f))