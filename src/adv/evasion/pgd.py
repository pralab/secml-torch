from torch.utils.data import DataLoader

from src.adv.backends import Backends
from src.adv.evasion.base_evasion_attack import BaseEvasionAttackCreator
from src.adv.evasion.threat_models import ThreatModels
from src.models.base_model import BaseModel


class PGD(BaseEvasionAttackCreator):
	def __init__(self, threat_model: str, epsilon: float, num_steps: int, step_size: float, random_start: bool,
	             backend: str = Backends.FOOLBOX, **kwargs):
		self.check_threat_model_available(threat_model)
		implementation = self.get_implementation(backend)
		return implementation(threat_model, epsilon, num_steps, step_size, random_start, kwargs)

	def __call__(self, model: BaseModel, data_loader: DataLoader) -> DataLoader:
		raise NotImplementedError("This class only generates the right PGD attack, it should not be used.")
