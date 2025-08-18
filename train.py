import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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
class SimpleAutoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Conv2d(3, 8, 3, stride=2, padding=1)
		self.relu = nn.ReLU()
		self.decoder = nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.encoder(x)
		x = self.relu(x)
		x = self.decoder(x)
		x = self.sigmoid(x)
		return x

def main():
	from_dir = './dataset/from'
	to_dir = './dataset/to'
	# Réduction de la taille des images pour limiter la mémoire
	transform = transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor()
	])
	dataset = ImagePairDataset(from_dir, to_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = SimpleAutoencoder().to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	epochs = 5
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		for from_img, to_img in dataloader:
			from_img = from_img.to(device)
			to_img = to_img.to(device)
			optimizer.zero_grad()
			output = model(from_img)
			loss = criterion(output, to_img)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * from_img.size(0)
		print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataset):.4f}")

	torch.save(model.state_dict(), 'image2image_model.pth')
	print("Modèle sauvegardé sous 'image2image_model.pth'.")

if __name__ == '__main__':
	main()
