import numpy as np
import subprocess

num_range = np.linspace(3, 68, 66, dtype=int)
add1 = '/home/zwenbo/Documents/research/deform/rope_dataset'
total = 0
list_file = []

for j in num_range:
    if len(str(j)) == 1:
        add2 = 'run0{}'.format(j)
    elif len(str(j)) == 2:
        add2 = 'run{}'.format(j)

    add = 'cd ' + add1 + '/rope_all/' + add2 + '; ls -1 | wc -l'
    num_file = int(subprocess.check_output(add, shell=True)) #- 1    
    list_file.append(num_file)
    total += num_file

print('image list:', list_file)
print('total number of files: {}'.format(total))