import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange

# Define the Vision Transformer model
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape

        # Split the embedding into self.num_heads pieces
        query = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads)
        key = rearrange(key, "b s (h d) -> b h s d", h=self.num_heads)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.num_heads)

        # Perform linear transformation and scale dot-product attention
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        # Apply softmax
        attention = torch.softmax(scores, dim=-1)

        # Attend to values
        out = torch.matmul(attention, value)

        # Reshape and concatenate attention heads
        out = rearrange(out, "b h s d -> b s (h d)")

        # Apply final linear transformation
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention layer
        attention_out = self.attention(x, x, x)
        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(attention_out))
        # MLP layer
        mlp_out = self.mlp(x)
        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(mlp_out))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        mlp_dim=1024,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # RGB channels
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Flatten image into patches
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        # Add positional embeddings
        x = torch.cat([x, self.positional_embedding.expand(batch_size, -1, -1)], dim=1)
        x = self.dropout(x)
        # Transformer layers
        x = self.transformer(x)
        # Layer normalization
        x = self.layer_norm(x)
        # Class token
        class_token = x[:, 0]
        # Classification layer
        x = self.fc(class_token)
        return x


    def forward_virtual(self, x):
        batch_size, _, _, _ = x.shape
        # Flatten image into patches
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        # Add positional embeddings
        x = torch.cat([x, self.positional_embedding.expand(batch_size, -1, -1)], dim=1)
        x = self.dropout(x)
        # Transformer layers
        x = self.transformer(x)
        # Layer normalization
        x = self.layer_norm(x)
        # Class token
        class_token = x[:, 0]
        # Classification layer
        x = self.fc(class_token)
        return x, class_token

    def get_fc(self):
        return self.fc


class VisionTransformer_cloud(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=30,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        mlp_dim=1024,
        dropout=0.1,
    ):
        super(VisionTransformer_cloud, self).__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # RGB channels
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Flatten image into patches
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        # Add positional embeddings
        x = torch.cat([x, self.positional_embedding.expand(batch_size, -1, -1)], dim=1)
        x = self.dropout(x)
        # Transformer layers
        x = self.transformer(x)
        # Layer normalization
        x = self.layer_norm(x)
        # Class token
        class_token = x[:, 0]
        # Classification layer
        x = self.fc(class_token)
        return x


    def forward_virtual(self, x):
        batch_size, _, _, _ = x.shape
        # Flatten image into patches
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embedding(x)
        # Add positional embeddings
        x = torch.cat([x, self.positional_embedding.expand(batch_size, -1, -1)], dim=1)
        x = self.dropout(x)
        # Transformer layers
        x = self.transformer(x)
        # Layer normalization
        x = self.layer_norm(x)
        # Class token
        class_token = x[:, 0]
        # Classification layer
        x = self.fc(class_token)
        return x, class_token

    def get_fc(self):
        return self.fc
#
# # Define transformations for data augmentation
# transform_train = transforms.Compose(
#     [
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )
#
#
#
# transform_test = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )
#
# # Load CIFAR-10 dataset
# trainset = torchvision.datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform_train
# )
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
#
# testset = torchvision.datasets.CIFAR10(
#     root="./data", train=False, download=True, transform=transform_test
# )
# testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
#
# # Initialize the Vision Transformer model
# model = VisionTransformer()
#
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Train the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# for epoch in range(10):  # Loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # Get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)
#
#         # Zero the parameter gradients
#         optimizer.zero_grad()
#
#         # Forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # Print statistics
#         running_loss += loss.item()
#         if i % 100 == 99:  # Print every 100 mini-batches
#             print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0
#
# print("Finished Training")
#
# # Test the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# # print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))
#
#
# import torch
# import torch.nn as nn
#
# # B -> Batch Size
# # C -> Number of Input Channels
# # IH -> Image Height
# # IW -> Image Width
# # P -> Patch Size
# # E -> Embedding Dimension
# # S -> Sequence Length = IH/P * IW/P
# # Q -> Query Sequence length
# # K -> Key Sequence length
# # V -> Value Sequence length (same as Key length)
# # H -> Number of heads
# # HE -> Head Embedding Dimension = E/H
#
#
# class EmbedLayer(nn.Module):
#     def __init__(self, n_channels, embed_dim, image_size, patch_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Pixel Encoding
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # Cls Token
#         self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)  # Positional Embedding
#
#     def forward(self, x):
#         x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
#         x = x.reshape([x.shape[0], x.shape[1], -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
#         x = x.transpose(1, 2)  # B E S -> B S E
#         x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
#         x = x + self.pos_embedding  # Adding positional embedding
#         return x
#
#
# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, n_attention_heads):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.n_attention_heads = n_attention_heads
#         self.head_embed_dim = embed_dim // n_attention_heads
#
#         self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
#         self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
#         self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
#
#     def forward(self, x):
#         b, s, e = x.shape
#
#         xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
#         xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
#         xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
#         xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
#         xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
#         xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE
#
#         # Compute Attention presoftmax values
#         xk = xk.transpose(-1, -2)  # B, H, K, HE -> B, H, HE, K
#         x_attention = torch.matmul(xq, xk)  # B, H, Q, HE  *  B, H, HE, K -> B, H, Q, K
#
#         # Scale presoftmax values for stability
#         x_attention /= float(self.head_embed_dim) ** 0.5
#
#         # Compute Attention Matrix
#         x_attention = torch.softmax(x_attention, dim=-1)
#
#         # Compute Attention Values
#         x = torch.matmul(x_attention, xv)  # B, H, Q, K * B, H, V, HE -> B, H, Q, HE
#
#         # Format the output
#         x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
#         x = x.reshape(b, s, e)  # B, Q, H, HE -> B, Q, E
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, embed_dim, n_attention_heads, forward_mul):
#         super().__init__()
#         self.attention = SelfAttention(embed_dim, n_attention_heads)
#         self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
#         self.activation = nn.GELU()
#         self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#
#     def forward(self, x):
#         x = x + self.attention(self.norm1(x)) # Skip connections
#         x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  # Skip connections
#         return x
#
#
# class Classifier(nn.Module):
#     def __init__(self, embed_dim, n_classes):
#         super().__init__()
#         # Newer architectures skip fc1 and activations and directly apply fc2.
#         self.fc1 = nn.Linear(embed_dim, embed_dim)
#         self.activation = nn.Tanh()
#         self.fc2 = nn.Linear(embed_dim, n_classes)
#
#     def forward(self, x):
#         x = x[:, 0, :]  # Get CLS token
#         x = self.fc1(x)
#         feature = self.activation(x)
#         x = self.fc2(x)
#         return x, feature
#
#
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self, n_channels=3, embed_dim=64, n_layers=6, n_attention_heads=4, forward_mul=2, image_size=32, patch_size=4, n_classes=10):
#         super().__init__()
#         self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
#         self.encoder = nn.Sequential(*[Encoder(embed_dim, n_attention_heads, forward_mul) for _ in range(n_layers)], nn.LayerNorm(embed_dim))
#         self.norm = nn.LayerNorm(embed_dim) # Final normalization layer after the last block
#         self.classifier = Classifier(embed_dim, n_classes)
#
#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.encoder(x)
#         x = self.norm(x)
#         x, feature = self.classifier(x)
#         return x
#
#     def forward_virtual(self, x):
#         x = self.embedding(x)
#         x = self.encoder(x)
#         x = self.norm(x)
#         x, feature = self.classifier(x)
#         return x, feature
#
#     def get_fc(self):
#         return self.classifier