import os
from subprocess import call

for i in range(69):
    if len(str(i)) == 1:
        # call('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_test', shell=True)
        call('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_interp_cubic; mkdir run0{}'.format(i), shell=True)
        # os.system('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_test')
        # os.system('mkdir run0{}'.format(i))
    elif len(str(i)) == 2:
        # call('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_test', shell=True)
        call('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_interp_cubic; mkdir run{}'.format(i), shell=True)        
        # os.system('cd /home/zwenbo/Documents/research/deform/rope_dataset/rope_test')
        # os.system('mkdir run{}'.format(i))    