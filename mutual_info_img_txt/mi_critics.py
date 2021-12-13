import torch

def dv_bound_loss(discriminator_logits, pos_size, device):
	size = discriminator_logits.shape[0]

	# The first pos_size elements are positive
	pos_energy = torch.mean(discriminator_logits[:pos_size])

	logsumexp = torch.logsumexp(discriminator_logits[pos_size:], dim=0)
	neg_energy = logsumexp - torch.log(torch.tensor(size-pos_size).float()).to(device)

	return neg_energy-pos_energy

def infonce_bound_loss(discriminator_logits, pos_size, device):
	size = discriminator_logits.shape[0]

	# The first pos_size elements are positive
	pos_energy = torch.mean(discriminator_logits[:pos_size])

	logsumexp = torch.logsumexp(discriminator_logits[pos_size:], dim=0)
	neg_energy = torch.mean(logsumexp)

	return neg_energy-pos_energy
