import os
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Dataset personnalisé pour charger les paires d'images
class ImagePairDataset(Dataset):
	def __init__(self, from_dir, to_dir, transform=None):
		self.from_dir = from_dir
		self.to_dir = to_dir
		self.transform = transform
		self.filenames = sorted(os.listdir(from_dir))
	def __len__(self):
		return len(self.filenames)
	def __getitem__(self, idx):
		from_img = Image.open(os.path.join(self.from_dir, self.filenames[idx])).convert('RGB')
		to_img = Image.open(os.path.join(self.to_dir, self.filenames[idx])).convert('RGB')
		if self.transform:
			from_img = self.transform(from_img)
			to_img = self.transform(to_img)
		return from_img, to_img

# Modèle simple de type autoencoder

# U-Net simplifié pour image2image

# Spatial Transformer Network pour les transformations géométriques
class SpatialTransformerNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		# Localization network pour prédire les paramètres de transformation
		self.localization = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.AdaptiveAvgPool2d((4, 4)),
		)
		self.fc_loc = nn.Sequential(
			nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
			nn.Linear(128, 6)  # 6 paramètres pour transformation affine
		)
		# Initialisation pour transformation identité
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def forward(self, x):
		# Calculer les paramètres de transformation
		loc_features = self.localization(x)
		loc_features = loc_features.view(-1, 128 * 4 * 4)
		theta = self.fc_loc(loc_features)
		theta = theta.view(-1, 2, 3)
		
		# Appliquer la transformation
		grid = F.affine_grid(theta, x.size(), align_corners=False)
		transformed = F.grid_sample(x, grid, align_corners=False)
		return transformed

# U-Net++ simplifié avec batch normalization et STN
class UNetPP(nn.Module):
	def __init__(self):
		super().__init__()
		# Spatial Transformer Network en entrée
		self.stn = SpatialTransformerNetwork()
		
		self.enc1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			nn.Dropout2d(0.1)
		)
		self.pool1 = nn.MaxPool2d(2)
		self.enc2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.Dropout2d(0.1)
		)
		self.pool2 = nn.MaxPool2d(2)
		self.enc3 = nn.Sequential(
			nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.Dropout2d(0.1)
		)
		self.pool3 = nn.MaxPool2d(2)
		self.enc4 = nn.Sequential(
			nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.Dropout2d(0.2)
		)
		self.pool4 = nn.MaxPool2d(2)
		# Niveau supplémentaire (bottleneck)
		self.enc5 = nn.Sequential(
			nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
			nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
			nn.Dropout2d(0.3)
		)
		
		# Décodeur avec niveau supplémentaire
		self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
		self.dec4 = nn.Sequential(
			nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.Dropout2d(0.2)
		)
		self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.dec3 = nn.Sequential(
			nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.Dropout2d(0.1)
		)
		self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.dec2 = nn.Sequential(
			nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.Dropout2d(0.1)
		)
		self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.dec1 = nn.Sequential(
			nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
		)
		self.final = nn.Conv2d(64, 3, 1)
		# Tanh au lieu de Sigmoid pour une meilleure gamme dynamique
		self.tanh = nn.Tanh()

	def forward(self, x):
		# Transformation géométrique d'abord
		x_transformed = self.stn(x)
		
		# Encodeur
		e1 = self.enc1(x_transformed)
		p1 = self.pool1(e1)
		e2 = self.enc2(p1)
		p2 = self.pool2(e2)
		e3 = self.enc3(p2)
		p3 = self.pool3(e3)
		e4 = self.enc4(p3)
		p4 = self.pool4(e4)
		e5 = self.enc5(p4)
		
		# Décodeur
		u4 = self.up4(e5)
		cat4 = torch.cat([u4, e4], dim=1)
		d4 = self.dec4(cat4)
		u3 = self.up3(d4)
		cat3 = torch.cat([u3, e3], dim=1)
		d3 = self.dec3(cat3)
		u2 = self.up2(d3)
		cat2 = torch.cat([u2, e2], dim=1)
		d2 = self.dec2(cat2)
		u1 = self.up1(d2)
		cat1 = torch.cat([u1, e1], dim=1)
		d1 = self.dec1(cat1)
		out = self.final(d1)
		out = self.tanh(out)
		return out

def main():
	import torch.backends.cudnn
	torch.backends.cudnn.benchmark = True
	from_dir = './dataset/from'
	to_dir = './dataset/to'
	# Réduction de la taille des images pour limiter la mémoire
	transform = transforms.Compose([
		transforms.ToTensor()
	])
	dataset = ImagePairDataset(from_dir, to_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if device.type == 'cuda':
		print("GPU détectée : entraînement sur CUDA.")
	else:
		print("GPU non détectée : entraînement sur CPU.")
	model = UNetPP().to(device)
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	# Sélection du modèle à charger dans le dossier 'modeles'
	import glob
	import re
	model_dir = 'modeles'
	os.makedirs(model_dir, exist_ok=True)
	checkpoints = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth')))
	print("\nModèles disponibles :")
	for i, ckpt in enumerate(checkpoints):
		print(f"[{i}] {ckpt}")
	print("[N] Nouveau modèle (entraînement à partir de zéro)")

	# Variables pour la reprise
	start_epoch = 0
	best_loss = float('inf')

	# Sélection automatique du dernier modèle existant
	last_ckpt = checkpoints[-1] if checkpoints else None
	if last_ckpt and os.path.exists(last_ckpt):
		print(f"\nReprise automatique du dernier modèle : {last_ckpt}")
		choix = input("Appuyez sur Entrée pour continuer, ou tapez N pour repartir d'un modèle vierge : ").strip()
		if choix.lower() == 'n':
			print("Entraînement à partir de zéro.")
		else:
			try:
				print(f"Chargement du modèle {last_ckpt}")
				checkpoint = torch.load(last_ckpt, map_location=device)
				
				# Compatibilité avec les anciens formats de checkpoint
				if 'model_state_dict' in checkpoint:
					# Nouveau format (checkpoint complet)
					model.load_state_dict(checkpoint['model_state_dict'])
					optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
					start_epoch = checkpoint['epoch']
					best_loss = checkpoint.get('loss', float('inf'))
					print(f"Reprise à l'époque {start_epoch + 1}/10000")
					print(f"Loss précédente : {best_loss:.4f}")
				else:
					# Ancien format (seulement state_dict)
					model.load_state_dict(checkpoint)
					# Extraire le numéro d'époque du nom de fichier si possible
					epoch_match = re.search(r'epoch[_]?(\d+)', last_ckpt)
					if epoch_match:
						start_epoch = int(epoch_match.group(1))
						print(f"Ancien format détecté. Reprise à l'époque {start_epoch + 1}/10000")
					else:
						print("Ancien format détecté. Reprise à l'époque 1/10000")
					print("Note: L'état de l'optimizer sera réinitialisé.")
			except Exception as e:
				print(f"Erreur lors du chargement du checkpoint {last_ckpt}: {e}")
				print("Le fichier semble corrompu. Suppression et recherche du checkpoint précédent...")
				os.remove(last_ckpt)
				# Rechercher le checkpoint précédent
				checkpoints = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth')))
				if checkpoints:
					last_ckpt = checkpoints[-1]
					print(f"Tentative avec le checkpoint précédent : {last_ckpt}")
					try:
						checkpoint = torch.load(last_ckpt, map_location=device)
						if 'model_state_dict' in checkpoint:
							model.load_state_dict(checkpoint['model_state_dict'])
							optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
							start_epoch = checkpoint['epoch']
							best_loss = checkpoint.get('loss', float('inf'))
							print(f"Reprise à l'époque {start_epoch + 1}/10000")
							print(f"Loss précédente : {best_loss:.4f}")
						else:
							model.load_state_dict(checkpoint)
							epoch_match = re.search(r'epoch[_]?(\d+)', last_ckpt)
							if epoch_match:
								start_epoch = int(epoch_match.group(1))
								print(f"Ancien format détecté. Reprise à l'époque {start_epoch + 1}/10000")
							else:
								print("Ancien format détecté. Reprise à l'époque 1/10000")
					except Exception as e2:
						print(f"Erreur également avec {last_ckpt}: {e2}")
						print("Démarrage d'un nouvel entraînement.")
				else:
					print("Aucun checkpoint valide trouvé. Démarrage d'un nouvel entraînement.")
	else:
		print("Aucun modèle trouvé, entraînement à partir de zéro.")

	# Loss functions améliorées pour rectification géométrique
	class PerceptualLoss(nn.Module):
		def __init__(self):
			super().__init__()
			vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
			for param in vgg.parameters():
				param.requires_grad = False
			self.vgg = vgg
		def forward(self, x, y):
			# x, y: [B, 3, H, W] in [-1,1] (Tanh output)
			# Normaliser pour VGG [0,1]
			x_norm = (x + 1) / 2
			y_norm = (y + 1) / 2
			# VGG expects [B, 3, H, W] normalisé
			mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
			std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
			x_vgg = (x_norm - mean) / std
			y_vgg = (y_norm - mean) / std
			return nn.functional.mse_loss(self.vgg(x_vgg), self.vgg(y_vgg))

	class SSIMLoss(nn.Module):
		def __init__(self, window_size=11):
			super().__init__()
			self.window_size = window_size
			
		def forward(self, x, y):
			# SSIM pour mesurer la similarité structurelle
			# x, y: [B, 3, H, W] in [-1,1]
			# Convertir en [0,1]
			x = (x + 1) / 2
			y = (y + 1) / 2
			
			mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size//2)
			mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size//2)
			
			mu_x_sq = mu_x ** 2
			mu_y_sq = mu_y ** 2
			mu_xy = mu_x * mu_y
			
			sigma_x_sq = F.avg_pool2d(x**2, self.window_size, stride=1, padding=self.window_size//2) - mu_x_sq
			sigma_y_sq = F.avg_pool2d(y**2, self.window_size, stride=1, padding=self.window_size//2) - mu_y_sq
			sigma_xy = F.avg_pool2d(x*y, self.window_size, stride=1, padding=self.window_size//2) - mu_xy
			
			C1 = 0.01 ** 2
			C2 = 0.03 ** 2
			
			ssim = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
			return 1 - ssim.mean()

	perceptual_criterion = PerceptualLoss().to(device)
	ssim_criterion = SSIMLoss().to(device)
	l1_criterion = nn.L1Loss()
	mse_criterion = nn.MSELoss()


	epochs = 10000  # ajustez selon vos besoins
	inter_dir = './dataset/inter'
	# Nettoyer le dossier inter avant de commencer
	if os.path.exists(inter_dir):
		for f in os.listdir(inter_dir):
			fp = os.path.join(inter_dir, f)
			if os.path.isfile(fp):
				os.remove(fp)
	os.makedirs(inter_dir, exist_ok=True)

	scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
	for epoch in range(start_epoch, epochs):
		model.train()
		running_loss = 0.0
		running_l1 = 0.0
		running_perc = 0.0
		running_ssim = 0.0
		for idx, (from_img, to_img) in enumerate(dataloader):
			from_img = from_img.to(device)
			to_img = to_img.to(device)
			
			# Normaliser les targets pour Tanh (de [0,1] vers [-1,1])
			to_img_tanh = to_img * 2.0 - 1.0
			
			optimizer.zero_grad()
			if device.type == 'cuda':
				with torch.amp.autocast('cuda'):
					output = model(from_img)
					# Loss combinée : L1 + Perceptual + SSIM
					loss_l1 = l1_criterion(output, to_img_tanh)
					loss_perc = perceptual_criterion(output, to_img_tanh)
					loss_ssim = ssim_criterion(output, to_img_tanh)
					loss = loss_l1 + 0.1 * loss_perc + 0.1 * loss_ssim
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
			else:
				output = model(from_img)
				loss_l1 = l1_criterion(output, to_img_tanh)
				loss_perc = perceptual_criterion(output, to_img_tanh)
				loss_ssim = ssim_criterion(output, to_img_tanh)
				loss = loss_l1 + 0.1 * loss_perc + 0.1 * loss_ssim
				loss.backward()
				optimizer.step()
			running_loss += loss.item() * from_img.size(0)
			running_l1 += loss_l1.item() * from_img.size(0)
			running_perc += loss_perc.item() * from_img.size(0)
			running_ssim += loss_ssim.item() * from_img.size(0)

			# Sauvegarde des images générées pour la première image de chaque époque
			if idx == 0:
				from torchvision.utils import save_image
				with torch.no_grad():
					# Reconvertir de [-1,1] vers [0,1] pour sauvegarde
					output_save = (output[0] + 1) / 2
					save_image(output_save.cpu(), os.path.join(inter_dir, f'epoch{epoch+1}_output.png'))
					save_image(from_img[0].cpu(), os.path.join(inter_dir, f'epoch{epoch+1}_input.png'))
					save_image(to_img[0].cpu(), os.path.join(inter_dir, f'epoch{epoch+1}_target.png'))

		print(f"Epoch {epoch+1}/{epochs}, Total Loss: {running_loss/len(dataset):.4f} | L1: {running_l1/len(dataset):.4f} | Perceptual: {running_perc/len(dataset):.4f} | SSIM: {running_ssim/len(dataset):.4f}")

		# Sauvegarde régulière du modèle dans 'modeles'
		current_loss = running_loss/len(dataset)
		if (epoch + 1) % 20 == 0:
			checkpoint = {
				'epoch': epoch + 1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': current_loss,
			}
			ckpt_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
			torch.save(checkpoint, ckpt_path)
			print(f"Checkpoint sauvegardé sous '{ckpt_path}'.")
			
			# Garder seulement les 3 derniers checkpoints
			checkpoints = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth')))
			if len(checkpoints) > 3:
				old_checkpoints = checkpoints[:-3]  # Tous sauf les 3 derniers
				for old_ckpt in old_checkpoints:
					os.remove(old_ckpt)
					print(f"Ancien checkpoint supprimé : {old_ckpt}")

	# Sauvegarde finale dans 'modeles'
	final_checkpoint = {
		'epoch': epochs,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': current_loss,
	}
	final_path = os.path.join(model_dir, f'checkpoint_epoch_{epochs:04d}_final.pth')
	torch.save(final_checkpoint, final_path)
	print(f"Checkpoint final sauvegardé sous '{final_path}'.")

if __name__ == '__main__':
	main()
