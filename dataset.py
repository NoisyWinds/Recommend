import itertools
from collections import defaultdict
from trainset import Trainset
import os
class Dataset(object):
	"""

	"""
	def __init__(self, reader):
		self.reader = reader

	@classmethod
	def load_from_file(self,file_path,reader):
		Dataset.__init__(self,reader)
		with open(os.path.expanduser(file_path)) as f:
			self.raw_ratings = [self.reader.parse_line(line) for line in itertools.islice(f, self.reader.skip_lines, None)]
			raw2inner_id_users = {}
			raw2inner_id_items = {}

			current_u_index = 0
			current_i_index = 0

			ur = defaultdict(list)
			ir = defaultdict(list)

			for urid, irid, r in self.raw_ratings:
				try:
					uid = raw2inner_id_users[urid]
				except KeyError:
					uid = current_u_index
					raw2inner_id_users[urid] = current_u_index
					current_u_index += 1
				try:
					iid = raw2inner_id_items[irid]
				except KeyError:
					iid = current_i_index
					raw2inner_id_items[irid] = current_i_index
					current_i_index += 1

				ur[uid].append((iid, r))
				ir[iid].append((uid, r))

			n_users = len(ur)  # number of users
			n_items = len(ir)  # number of items
			n_ratings = len(self.raw_ratings)

			trainset = Trainset(ur,
								ir,
								n_users,
								n_items,
								n_ratings,
								self.reader.rating_scale,
								self.reader.offset,
								raw2inner_id_users,
								raw2inner_id_items)
			return trainset