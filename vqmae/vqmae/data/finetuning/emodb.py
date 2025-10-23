import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

class Emodb:
    emotion_map = {
        "W": "anger",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happiness",
        "T": "sadness",
        "N": "neutral"
    }
    
    def __init__(self, root: Path = None, ext="wav"):
        self.root = Path(root)
        self.length = len(glob.glob(f"{root}/**/*.{ext}"))
        self.table = None

    def generator(self):
        for file_path in self.root.glob("*.wav"):
            name = file_path.name
            # Skip hidden/system files
            if name.startswith('.'):
                continue
            try:
                speaker_id = name[:2]  # first 2 chars
                emotion_letter = name[5]  # 6th position
                emotion_str = self.emotion_map.get(emotion_letter, None)
                if emotion_str is None:
                    continue
                emotion_idx = list(self.emotion_map.keys()).index(emotion_letter)
            except Exception:
                continue
            yield speaker_id, name, file_path, emotion_idx

    def generate_table(self):
        files_list = []
        id_list = []
        name_list = []
        emotion_list = []

        with tqdm(total=self.length, desc="Create table (EMO-DB)") as pbar:
            for speaker_id, name, path, emotion_idx in self.generator():
                files_list.append(path)
                id_list.append(speaker_id)
                name_list.append(name)
                emotion_list.append(emotion_idx)
                pbar.update(1)

        self.table = pd.DataFrame(
            np.array([id_list, name_list, files_list, emotion_list]).T,
            columns=['id', 'name', 'path', 'emotion']
        )

if __name__ == "__main__":
    dataset = Emodb(root=Path("emodb/wav"))
    dataset.generate_table()
    print(dataset.table.head())
