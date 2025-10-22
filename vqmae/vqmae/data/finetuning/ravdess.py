import os
from tqdm import tqdm
import glob
import pandas
import numpy as np
from pathlib import Path


class Ravdess:
    def __init__(self, root: Path = None, ext="wav"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/*.{ext}"))
        self.table = None

    @staticmethod
    def __generator__(directory: Path):
        all_entries = os.listdir(directory)
        for entry in all_entries:
            # Skip hidden/system files like .DS_Store
            if str(entry).startswith('.'):
                continue
            yield entry, directory / entry

    def generator(self):
        for id, id_root in self.__generator__(self.root):
            # Top-level should be actor directories; skip files like .DS_Store
            if not id_root.is_dir():
                continue
            for name, path in self.__generator__(id_root):
                # Only keep audio files with expected extension
                if not path.is_file() or path.suffix.lower() != f".{ 'wav' }":
                    continue
                # RAVDESS filename convention: XX-XX-XX-XX-XX-XX-XX.wav
                # Emotion index is the 3rd field (1-based), map to [0..]
                try:
                    emotion = int(name.split(".")[0].split("-")[2]) - 1
                except Exception:
                    # Skip files not matching expected naming convention
                    continue
                yield id, name, path, emotion

    def generate_table(self):
        files_list = []
        id_list = []
        name_list = []
        emotion_list = []
        with tqdm(total=self.length, desc=f"Create table (RAVDESS): ") as pbar:
            for id, name, path, emotion in self.generator():
                files_list.append(path)
                id_list.append(id)
                emotion_list.append(emotion)
                name_list.append(name)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([id_list, name_list,  files_list, emotion_list]).transpose(),
                                      columns=['id', 'name', 'path', 'emotion'])


if __name__ == '__main__':
    vox = Ravdess(root=Path(r"D:\These\data\Audio\RAVDESS"))
    vox.generate_table()
    print(vox.table["id"])

