class Antigen():
	def __init__(self, image_array):
		self.image_array = image_array


class Limfocit_B():
	"""
	ab = [(a1, b1),(a2, b2),(a3, b3),(a4, b4)]
	(an, bn) - coordinates of white pixels
	cd = [(c1, d1),(c2, d2),(c3, d3),(c4, d4)]
	(cn, dn) - coordinates of black pixels
	S = S1 + S2
	S1 = len(ab)
	S2 = len(cd)
	"""
	def __init__(self, S, limfocit_id):
		self.S = S
		self.id = limfocit_id
		self.last_activity_time = None

	def affinity(self, antigen):
		P = 0
		for i in antigen.ab:
			if i in self.ab:
				 P += 1
		affinity = P/(len(self.ab) + len(self.cd))
		return P
			
	def is_reacting(self, affinity):
		return affinity >= self.S 


# class Antibody():
# 	pass

