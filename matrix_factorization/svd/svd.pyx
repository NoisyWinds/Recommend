import numpy as np
cimport numpy as np
import random
from prediction import Prediction
from prediction import PredictionImpossible
class SVD(object):
	"""
	这个著名的SVD算法是基于2006年 Netflix Prize 中 Simon Funk 在博客公开的算法
	`<http://sifter.org/~simon/journal/20061211.html>`
	被称为（Funk-SVD）。

	The prediction: math:`\\hat{r}_{ui}` is set as:

	.. math::
		\hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

	当 user :math:`u` unknown的情况下, 偏移 :math:`b_u` 和 factors
	:math:`p_u` 被设为0。同样，当 item :math:`i` unknown的时候 
	:math:`b_i` 和 :math:`q_i` 也被设为了0。

	详情可参见 Netflix Prize `Koren:2009` `Ricci:2010`

	所以我们需要预估所有的unknown，调整最小的损失：

	.. math::
		\sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
		\lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)

	我们使用一个简单的随机梯度下降来最小化损失：

	.. math::
		b_u &\\leftarrow b_u &+ \gamma (e_{ui} - \lambda b_u;\\\\
		b_i &\\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\\\
		p_u &\\leftarrow p_u &+ \gamma (e_{ui} \\cdot q_i - \lambda p_u)\\\\
		q_i &\\leftarrow q_i &+ \gamma (e_{ui} \\cdot p_u - \lambda q_i)

	:math:`e_{ui} = r_{ui} - \\hat{r}_{ui}` 这一步完成对所有 rating 的训练并且迭代``n_epochs``次。
	初始化参数为``0``。user 和 item factors 初始化为随机正态分布。这里可以调整参数``init_mean`` 和 ``init_std_dev``

	你也可以控制学习率 :math `\gamma` 和正则项 :math:`\lambda` 
	默认情况下 rates 是 ``0.005`` regularization terms 是 ``0.02``

	无偏移版本（unbiased）：

		你可以使用这个算法的无偏移版本进行预测
		math::
				\hat{r}_{ui} = q_i^Tp_u

		这相当于概率矩阵分解
		设置 ``biased`` 参数为 ``False``来实现


	Args:
		n_factors: The number of factors. 默认是 is ``100``.
		n_epochs: 随机梯度下降迭代次数. 默认是 ``20``.
		biased(bool): 是否使用偏移. 默认是 is ``True``.
		init_mean: 初始化 factors 的正态分布的均值. 默认是 is ``0``.
		init_std_dev:factor 向量初始化的时候正态分布的标准偏差. 默认是 is ``0.1``.
		lr_all: 所有参数的学习率. 默认是 ``0.005``.
		reg_all: 所有参数的正则化. 默认是 is ``0.02``.
		lr_bu: 学习率 :math:`b_u`. 优先于 lr_all. 默认 ``None``. ``None`` 则会使用 lr_all
		lr_bi: 学习率 :math:`b_i`. 优先于 lr_all. 默认 ``None``. ``None`` 则会使用 lr_all
		lr_pu: 学习率 :math:`p_u`. 优先于 lr_all. 默认 ``None``. ``None`` 则会使用 lr_all
		lr_qi: 学习率 :math:`q_i`. 优先于 lr_all. 默认 ``None``. ``None`` 则会使用 lr_all
		reg_bu: 正则化项 :math:`b_u`. 优先于 reg_all. 默认 ``None``. ``None`` 则会使用 reg_all
		reg_bi: 正则化项 :math:`b_i`. 优先于 reg_all. 默认 ``None``. ``None`` 则会使用 reg_all
		reg_pu: 正则化项 :math:`p_u`. 优先于 reg_all. 默认 ``None``. ``None`` 则会使用 reg_all
		reg_qi: 正则化项 :math:`q_i`. 优先于 reg_all. 默认 ``None``. ``None`` 则会使用 reg_all
		verbose: 是否每次迭代都进行打印（方便调试）. 默认 ``False``.


	Attributes:
		pu(numpy array of size (n_users, n_factors)): The user factors (only
			exists if ``fit()`` has been called)
		qi(numpy array of size (n_items, n_factors)): The item factors (only
			exists if ``fit()`` has been called)
		bu(numpy array of size (n_users)): The user biases (only
			exists if ``fit()`` has been called)
		bi(numpy array of size (n_items)): The item biases (only
			exists if ``fit()`` has been called)
	"""
	def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
				 init_std_dev=.1, lr_all=.005,
				 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
				 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
				 verbose=False):

		self.n_factors = n_factors
		self.n_epochs = n_epochs
		self.biased = biased
		self.init_mean = init_mean
		self.init_std_dev = init_std_dev
		self.lr_bu = lr_bu if lr_bu is not None else lr_all
		self.lr_bi = lr_bi if lr_bi is not None else lr_all
		self.lr_pu = lr_pu if lr_pu is not None else lr_all
		self.lr_qi = lr_qi if lr_qi is not None else lr_all
		self.reg_bu = reg_bu if reg_bu is not None else reg_all
		self.reg_bi = reg_bi if reg_bi is not None else reg_all
		self.reg_pu = reg_pu if reg_pu is not None else reg_all
		self.reg_qi = reg_qi if reg_qi is not None else reg_all
		self.verbose = verbose

	def fit(self, trainset):
		self.trainset = trainset
		self.sgd(trainset)
		return self
		
	def sgd(self, trainset):

		"""
		这里是算法的核心部分，使用cython提升效率，需要通过 setup.py 编译后才能使用
		"""
		cdef np.ndarray[np.double_t] bu
		# item biases
		cdef np.ndarray[np.double_t] bi
		# user factors
		cdef np.ndarray[np.double_t, ndim=2] pu
		# item factors
		cdef np.ndarray[np.double_t, ndim=2] qi

		cdef int u, i, f
		cdef double r, err, dot, puf, qif
		cdef double global_mean = self.trainset.global_mean()

		cdef double lr_bu = self.lr_bu
		cdef double lr_bi = self.lr_bi
		cdef double lr_pu = self.lr_pu
		cdef double lr_qi = self.lr_qi

		cdef double reg_bu = self.reg_bu
		cdef double reg_bi = self.reg_bi
		cdef double reg_pu = self.reg_pu
		cdef double reg_qi = self.reg_qi

		#  random samples from a normal (Gaussian) distribution.
		rng = np.random.mtrand._rand

		bu = np.zeros(trainset.n_users, np.double)
		bi = np.zeros(trainset.n_items, np.double)
		pu = rng.normal(self.init_mean, self.init_std_dev,
						(trainset.n_users, self.n_factors))
		qi = rng.normal(self.init_mean, self.init_std_dev,
						(trainset.n_items, self.n_factors))

		if not self.biased:
			global_mean = 0

		for current_epoch in range(self.n_epochs):
			if self.verbose:
				print("Processing epoch {}".format(current_epoch))

			for u, i, r in trainset.all_ratings():

				# compute current error
				dot = 0  # <q_i, p_u>
				for f in range(self.n_factors):
					dot += qi[i, f] * pu[u, f]
				err = r - (global_mean + bu[u] + bi[i] + dot)

				# update biases
				if self.biased:
					bu[u] += lr_bu * (err - reg_bu * bu[u])
					bi[i] += lr_bi * (err - reg_bi * bi[i])

				# update factors
				for f in range(self.n_factors):
					puf = pu[u, f]
					qif = qi[i, f]
					pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
					qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

		self.bu = bu
		self.bi = bi
		self.pu = pu
		self.qi = qi

	def estimate(self, u, i):
		known_user = self.trainset.knows_user(u)
		known_item = self.trainset.knows_item(i)
		if self.biased:
			est = self.trainset.global_mean()

			if known_user:
				est += self.bu[u]

			if known_item:
				est += self.bi[i]

			if known_user and known_item:
				est += np.dot(self.qi[i], self.pu[u])

		else:
			if known_user and known_item:
				est = np.dot(self.qi[i], self.pu[u])
			else:
				raise ValueError('User and item are unkown.')
		return est

	def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
		'''
		用给定的用户和物品进行预测

		Args:
			uid: (Raw) id of the user
			iid: (Raw) id of the item
			r_ui(float): rule rating :math:`r_{ui}`. Optional, 默认 ``None``.
			clip(bool): Whether to clip the estimation into the rating scale.
				For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
				rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
				set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
				Default is ``True``.
			verbose(bool): 是否每次预测都进行打印（方便调试）. 默认 ``False``.
		'''
		try:
			iuid = self.trainset.to_inner_uid(uid)
		except ValueError:
			iuid = 'unknown__' + str(uid)
		try:
			iiid = self.trainset.to_inner_iid(iid)
		except ValueError:
			iiid = 'unknown__' + str(iid)

		details = {}
		try:
			est = self.estimate(iuid, iiid)
			if isinstance(est, tuple):
				est, details = est
			details['was_impossible'] = False

		except PredictionImpossible as e:
			est = self.trainset.global_mean()
			details['was_impossible'] = True
			details['reason'] = str(e)

		est -= self.trainset.offset

		if clip:
			lower_bound, higher_bound = self.trainset.rating_scale
			est = min(higher_bound, est)
			est = max(lower_bound, est)

		pred = Prediction(uid, iid, r_ui, est, details)

		if verbose:
			print(pred)
		return pred