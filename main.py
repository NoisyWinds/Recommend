from dataset import Dataset
from reader import Reader
from matrix_factorization.svd.svd import SVD
import os
reader = Reader(sep=",")
trainset = Dataset.load_from_file('train.txt',reader = reader)
algo = SVD()
algo.fit(trainset)
result = []
with open(os.path.expanduser('test.txt')) as f:
	test = f.read().splitlines()
	for item in test[1:]:
		item = item.split(',')
		# tuple prediction ('uid', 'iid', 'r_ui', 'est', 'details')
		predict = algo.predict(item[0],item[1])[3]
		result.append(str(int(round(predict))))

with open('result.txt','w') as f:
	f.write(''.join(result))
