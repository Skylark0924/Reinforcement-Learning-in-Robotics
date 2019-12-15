import os
import shutil

root_dir = '/home/skylark/PycharmRemote/ldf_download'
new_dir = '/home/skylark/PycharmRemote/NeurIPS'
if not os.path.exists(new_dir):
    os.mkdir(new_dir)

files = os.listdir(root_dir)

keywords = ['einforcement', 'obot', 'RL', 'agent', 'Meta', 'policy', 'actor']

for file in files:
    if any(keyword in file for keyword in keywords):
        shutil.copy(os.path.join(root_dir, file), new_dir)
