class ThreatModels:
	L0 = "l0"
	L1 = "l1"
	L2 = "l2"
	LINF = "linf"

	@classmethod
	def is_threat_model_available(cls, threat_model):
		is_available = getattr(ThreatModels, threat_model, None)
		return is_available is not None
