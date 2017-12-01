from megskull.opr.loss import WeightDecay
from megskull.utils.meta import override
from megskull.utils.logconf import get_logger
import megbrain as mgb
import collections
import re
import fnmatch

logger = get_logger(__name__)

class MyWeightDecay(WeightDecay):
	@override(WeightDecay)
	def _init_decay_weights(self, decay, params):
		"""init param decay weights and store it in :attr:`_param_weights`

		:return: list of :class:`.ParamProvider` for the params to be decayed
		"""
		if isinstance(decay, dict):
			decay = decay.items()
		assert isinstance(decay, collections.Iterable), ('invalid decay description: {}'.format(decay))
		# list of list [re, weight, orig_name, used]
		re_weight_used = []
		#self._orig_decay_spec = []
		for name, val in decay:
			#self._orig_decay_spec.append((name, val))
			assert isinstance(name, str), (
				'invalid decay name: {}'.format(name))
			name_re = re.compile(fnmatch.translate(name), flags=re.IGNORECASE)
			re_weight_used.append([name_re, float(val) * 0.5, name, False])
		#self._orig_decay_spec = tuple(self._orig_decay_spec)
		
		used_params = []
		self._param_weights = []
		for param in params:
			for item in re_weight_used:
				if not item[0].match(param.name):
					continue
				item[-1] = True
				used_params.append(param)
				self._param_weights.append(mgb.SharedScalar(item[1]))
				break # use first match
	
		for i in re_weight_used:
			if not i[-1]:
				logger.warning('unused weight decay spec: {}'.format(i[2]))
		return used_params
		"""
		new_decay = {}
		for i in decay.items():
			k = i[0]
			it = mgb.SharedScalar(i[1])
			new_decay[k] = it
		super()._init_decay_weights(new_decay, params)
		"""
	
	@override(WeightDecay)
	def _init_output_mgbvar(self, env):
		loss = env.get_mgbvar(self._orig_loss)
		log_msg = ['weight decay:']
		for pvar, weight in zip(self._params, self._param_weights):
			log_msg.append(' {}: {}'.format(pvar.name, weight))

			cur = (env.get_mgbvar(pvar) ** 2).sum() * (weight)
			if cur.comp_node != loss.comp_node:
				cur = mgb.opr.copy(cur, comp_node=loss.comp_node)
			loss += cur
		env.set_mgbvar(self._var_output, loss)
		if len(log_msg) > 1 and env.flags.verbose_fprop:
			logger.info('\n'.join(log_msg))
	
	def Mul_Wc(self, rate):
		for i in self._param_weights:
			i.set(float(i.get()) * float(rate))

