
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Dataset personnalisé (repris de train.py)
class ImagePairDataset(torch.utils.data.Dataset):
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
		return from_img, to_img, self.filenames[idx]



# Spatial Transformer Network (identique à train.py)
class SpatialTransformerNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.localization = nn.Sequential(
			nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
			nn.AdaptiveAvgPool2d((4, 4)),
		)
		self.fc_loc = nn.Sequential(
			nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
			nn.Linear(128, 6)
		)
		self.fc_loc[2].weight.data.zero_()
		self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

	def forward(self, x):
		loc_features = self.localization(x)
		loc_features = loc_features.view(-1, 128 * 4 * 4)
		theta = self.fc_loc(loc_features)
		theta = theta.view(-1, 2, 3)
		grid = F.affine_grid(theta, x.size(), align_corners=False)
		transformed = F.grid_sample(x, grid, align_corners=False)
		return transformed

# UNetPP identique à train.py
class UNetPP(nn.Module):
	def __init__(self):
		super().__init__()
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
		self.enc5 = nn.Sequential(
			nn.Conv2d(512, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
			nn.Conv2d(1024, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
			nn.Dropout2d(0.3)
		)
		
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
		self.tanh = nn.Tanh()

	def forward(self, x):
		x_transformed = self.stn(x)
		e1 = self.enc1(x_transformed)
		p1 = self.pool1(e1)
		e2 = self.enc2(p1)
		p2 = self.pool2(e2)
		e3 = self.enc3(p2)
		p3 = self.pool3(e3)
		e4 = self.enc4(p3)
		p4 = self.pool4(e4)
		e5 = self.enc5(p4)
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

def save_images(input_img, output_img, target_img, filename, results_dir):
	# Convertir les tenseurs en images PIL
	to_pil = transforms.ToPILImage()
	inp_pil = to_pil(input_img.cpu())
	# Convertir la sortie Tanh [-1,1] vers [0,1] pour sauvegarde
	output_norm = (output_img.detach().cpu() + 1) / 2
	out_pil = to_pil(output_norm)
	tgt_pil = to_pil(target_img.cpu())
	# Enregistrer les images
	if isinstance(filename, (tuple, list)):
		filename = filename[0]
	clean_filename = str(filename).replace("',)","").replace("[","").replace("]","").replace("'","")
	if not clean_filename.lower().endswith('.png') and not clean_filename.lower().endswith('.jpg'):
		clean_filename += '.png'
	inp_pil.save(os.path.join(results_dir, f"input_{clean_filename}"))
	out_pil.save(os.path.join(results_dir, f"output_{clean_filename}"))
	tgt_pil.save(os.path.join(results_dir, f"target_{clean_filename}"))

def main():
	from_dir = './dataset/from'
	to_dir = './dataset/to'
	transform = transforms.Compose([
		transforms.ToTensor()
	])
	dataset = ImagePairDataset(from_dir, to_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = UNetPP().to(device)
	# Charger le dernier checkpoint disponible dans modeles
	import glob
	model_dir = 'modeles'
	checkpoints = sorted(glob.glob(os.path.join(model_dir, 'checkpoint_epoch_*.pth')))
	last_ckpt = checkpoints[-1] if checkpoints else None
	if last_ckpt and os.path.exists(last_ckpt):
		print(f"Chargement du modèle : {last_ckpt}")
		checkpoint = torch.load(last_ckpt, map_location=device)
		
		# Compatibilité avec les anciens formats de checkpoint
		if 'model_state_dict' in checkpoint:
			# Nouveau format (checkpoint complet)
			model.load_state_dict(checkpoint['model_state_dict'])
			epoch_num = checkpoint['epoch']
			loss_val = checkpoint.get('loss', 'N/A')
			print(f"Modèle entraîné jusqu'à l'époque {epoch_num}, Loss: {loss_val}")
		else:
			# Ancien format (seulement state_dict)
			model.load_state_dict(checkpoint)
			print("Ancien format de checkpoint détecté.")
	else:
		print("Aucun modèle trouvé dans 'modeles', le modèle sera aléatoire.")
	model.eval()

	criterion = nn.L1Loss()
	total_loss = 0.0


	results_dir = './dataset/results'
	os.makedirs(results_dir, exist_ok=True)

	for from_img, to_img, filename in dataloader:
		from_img = from_img.to(device)
		to_img = to_img.to(device)
		# Convertir target en [-1,1] pour comparaison avec Tanh
		to_img_tanh = to_img * 2.0 - 1.0
		with torch.no_grad():
			output = model(from_img)
			loss = criterion(output, to_img_tanh)
		total_loss += loss.item()
		# Sauvegarde des images générées
		save_images(from_img[0], output[0], to_img[0], filename, results_dir)
		print(f"Fichier: {filename}, L1 Loss: {loss.item():.4f}")

	print(f"L1 Loss moyenne sur le dataset: {total_loss/len(dataset):.4f}")
	print("Note: Le modèle utilise Tanh [-1,1] et inclut un Spatial Transformer Network pour les corrections géométriques.")

if __name__ == '__main__':
	main()