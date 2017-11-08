import urllib.request
import os
import tarfile


def download_dataset(download_path, save_path, filename):
    file_path = save_path + filename

    print("Downloading dataset")
    urllib.request.urlretrieve(download_path, file_path)

    print("Extrating dataset")
    tar = tarfile.open(file_path)
    tar.extractall(path=save_path)
    tar.close()

    os.remove(file_path)

if __name__ == '__main__':
    download_dataset()
