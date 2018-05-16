'''This module contains the Reader class.'''
class Reader():
	def __init__(self,line_format='user item rating', sep=None,
				 rating_scale=(1, 5), skip_lines=0):
		self.sep = sep
		self.skip_lines = skip_lines
		self.rating_scale = rating_scale
		lower_bound,higher_bound = rating_scale
		self.offset = -lower_bound + 1 if lower_bound <= 0 else 0
		splitted_format = line_format.split()
		entities = ['user','item','rating']
		if any(field not in entities for field in splitted_format):
			raise ValueError('line_format parameter is incorrect.')
		self.indexes = [splitted_format.index(entity) for entity in entities]

	def parse_line(self, line):
		'''Parse a line.
		Ratings are translated so that they are all strictly positive.
		Args:
			line(str): The line to parse
		Returns:
			tuple: User id, item id, rating and timestamp. The timestamp is set
			to ``None`` if it does no exist.
			'''
		try:
			line = line.split(self.sep)
			uid, iid, r = (line[i].strip() for i in self.indexes)
		except IndexError:
			raise ValueError('Impossible to parse line. Check the line_format'
							 ' and sep parameters.')
		return uid, iid, float(r) + self.offset



