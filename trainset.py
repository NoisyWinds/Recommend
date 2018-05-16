import numpy as np
from six import iteritems
class Trainset:
	"""
	"""
	def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
				 offset, raw2inner_id_users, raw2inner_id_items):

		self.ur = ur
		self.ir = ir
		self.n_users = n_users
		self.n_items = n_items
		self.n_ratings = n_ratings
		self.rating_scale = rating_scale
		self.offset = offset
		self._raw2inner_id_users = raw2inner_id_users
		self._raw2inner_id_items = raw2inner_id_items
		self._global_mean = None


	def knows_user(self,uid):
		return uid in self.ur

	def knows_item(self,iid):
		return iid in self.ir

	def global_mean(self):
		# global_mean: The mean of all ratings :math:`\\mu`.
		if self._global_mean == None:
			self._global_mean = np.mean([r for (_,_,r) in self.all_ratings()])
		return self._global_mean

	def to_inner_uid(self, ruid):
		try:
			return self._raw2inner_id_users[ruid]
		except KeyError:
			raise ValueError('User ' + str(ruid) + ' is not part of the trainset.')
			
	def to_inner_iid(self, riid):
		try:
			return self._raw2inner_id_items[riid]
		except KeyError:
			raise ValueError('Item ' + str(riid) + ' is not part of the trainset.')
	def all_ratings(self):
		for u, u_ratings in iteritems(self.ur):
			for i, r in u_ratings:
				yield u, i, r