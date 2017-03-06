import subprocess
directories = ['coupa', 'bookstore', 'death', 'gates']
from sys import argv
num = int(argv[1])
directory = directories[num]
print directory
ps = []
for ct in range(4):
    subprocess.Popen(["./theirs", "{}/{}".format(directory, ct),">", "{}/{}/out.txt".format(directory, ct)])
p = subprocess.Popen(["./theirs", "{}/{}".format(directory, 4),">", "{}/{}/out.txt".format(directory, ct)])
p.wait()
