class ThreatModels:
	L0 = "l0"
	L1 = "l1"
	L2 = "l2"
	LINF = "linf"

	@classmethod
	def is_threat_model_available(cls, threat_model):
		return threat_model in (cls.L0, cls.L1, cls.L2, cls.LINF)
