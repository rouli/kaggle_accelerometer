class TimeIndex():
	def __init__(self, seq):
		"""
			sequence is made of tuples (timestamp, value)
		"""
		self.seq = sorted(seq)
		self.binsize = len(seq)/10
		percentiles = [self.seq[i][0] for i in range(0, len(seq), self.binsize)]+[self.seq[-1][0]+1]
		self.bins = [(i, start, stop) for i, (start, stop) in enumerate(zip(percentiles, percentiles[1:]))]

	def __getitem__(self, key):
		if isinstance(key, slice):
			indexes = [bin_index for bin_index, bin_start, bin_stop in self.bins if (key.start<=bin_start and key.stop>=bin_stop) or (bin_start<=key.start<=bin_stop) or (bin_start<=key.stop<=bin_stop)]
			if not indexes:
				return []
			return [(ts, val) for ts, val in self.seq[indexes[0]*self.binsize:(indexes[-1]+1)*self.binsize] if key.start<=ts<=key.stop]

		else:
			return self.seq[key]

	def __len__(self):
		return len(self.seq)