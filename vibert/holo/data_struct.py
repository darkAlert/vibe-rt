import os

class DSNode():
	def __init__(self, nodes, idx_start, idx_stop, leaf=False):
		# self.nodes = nodes
		self._idx_start = idx_start
		self._idx_stop = idx_stop
		self._leaf = leaf

	@property
	def idx_start(self):
		return self._idx_start

	@property
	def idx_stop(self):
		return self._idx_stop

	@property
	def is_leaf(self):
		return self._leaf


class DSLeaf():
	def __init__(self, root, name, sub_dir):
		self._root = root
		self._name = name
		self._dir = sub_dir

	@property
	def name(self):
		return self._name

	@property
	def dir(self):
		return self._dir

	@property
	def filename(self):
		return self.name + self._root.ext

	@property
	def path(self):
		return os.path.join(self.dir,self.filename)

	@property
	def abs_dir(self):
		return os.path.join(self._root.root_dir,self.dir)

	@property
	def abs_path(self):
		return os.path.join(self.abs_dir,self.filename)


class DataStruct():
	def __init__(self):
		self._root_dir = None
		self._ext = None
		self._level_names = None
		self._levels = None
		self._tree = None
		self._items = None

	def parse(self, root, levels, ext='.mp4'):
		assert os.path.exists(root)
		self._root_dir = root
		self._ext = ext if ext[0] == '.' else '.' + ext
		if levels[0] == '/':
			levels = levels[1:]
		if levels[-1] == '/':
			levels = levels[:-1]
		self._level_names = levels.split('/')
		self._levels = {l:[] for l in self._level_names}
		self._items = []
		self._tree = self.__parse__()

		return self

	@property
	def root_dir(self):
		return self._root_dir

	@property
	def ext(self):
		return self._ext

	def __len__(self):
		return len(self._items)

	def __parse__(self, path='', level=0):
		abs_path = os.path.join(self.root_dir, path)

		# Create leafs:
		if level == len(self._level_names):
			names = []
			for file in os.listdir(abs_path):
				if file.endswith(self.ext):
					names.append(file.split('.')[0])
			names.sort()
			idx = len(self._items)
			for name in names:
				self._items.append(DSLeaf(root=self, name=name, sub_dir=path))

			return DSNode(None, idx_start=idx, idx_stop=len(self._items), leaf=True)

		# Or go dipper in subdirs:
		subdirs = [p.name for p in os.scandir(abs_path) if p.is_dir()]
		subdirs.sort()
		nodes = {}
		idx = len(self._items)
		for dir_name in subdirs:
			sub_path = os.path.join(path,dir_name)
			node = self.__parse__(sub_path, level=level+1)
			nodes[dir_name] = node
			self._levels[self._level_names[level]].append((nodes[dir_name],sub_path))

		return DSNode(nodes, idx_start=idx, idx_stop=len(self._items))


	def items(self, start=0, stop=None, step=1):
		if isinstance(start, tuple):
			assert len(start)==2
			start, stop = start[0], start[1]
		elif isinstance(start, DSNode):
			start, stop = start.idx_start, start.idx_stop

		if stop is None:
			stop = len(self._items)

		for idx in range(start, stop, step):
			yield self._items[idx]


	def levels(self,name=None):
		if name is None:
			name = self._level_names[0]

		level = self._levels[name]
		for node,path in level:
			yield (node.idx_start, node.idx_stop),path

	def nodes(self, name):
		level = self._levels[name]
		for node,path in level:
			yield node, path
			

if __name__ == "__main__":
	root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/mp4'
	data = DataStruct().parse(root_dir, levels='subject/light/garment/scene', ext='mp4')

	# for item in data.items():
	# 	print (item.abs_path)

	for idx_range,path in data.levels('scene'):
		print ('====================:',path)
		for item in data.items(idx_range):
			print (item.abs_path)
