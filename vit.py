import torch 
from torch import nn

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=self.patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)


class ViT(nn.Module):
  def __init__(self, img_size=224, patch_size=16, embedding_dim=768, dropout=0.1, mlp_size=3072, num_transformer_layers = 12, num_heads=12, num_channels=3, num_classes=1000):
    super().__init__()
    # print("hi")

    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    # patch embedding
    self.patch_embedding = PatchEmbedding(in_channels=num_channels, patch_size=patch_size, embedding_dim=embedding_dim)

    # class token, requires_grad makes it learnable
    self.class_token = nn.Parameter(torch.randn(1,1,embedding_dim), requires_grad=True)

    # positional embedding
    num_patches = (img_size * img_size // patch_size**2) # N=HW/P^2
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

    # patch and positional embedding dropout
    self.embedding_dropout = nn.Dropout(p=dropout)

    # Single transformer encoder layer
    self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=mlp_size, activation="gelu", batch_first=True,
    norm_first=True)

    # Stack of Transformer Encoder Layers
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=mlp_size, activation="gelu", batch_first=True,
                                                                                              norm_first=True), num_layers=12)

    # MLP head
    self.mlp_head = nn.Sequential(
      #get info about layer norm from papers w/ code
      nn.LayerNorm(normalized_shape=embedding_dim),
      nn.Linear(in_features=embedding_dim, out_features=num_classes)
    )


  def forward(self, x):
    batch_size = x.shape[0]

    # Patch embedding
    x = self.patch_embedding(x)

    # Expand the class token across all the batch size and infer the dimension 
    class_token = self.class_token.expand(batch_size, -1, -1)

    # Prepend class token to patch embedding
    x = torch.cat((class_token, x), dim=1)

    # add pos_emb to class_token
    x = self.positional_embedding + x
    print(x.shape)

    # Dropout Add
    x = self.embedding_dropout(x)

    # Pass embedding through transformer stack
    x = self.transformer_encoder(x)

    # Pass first token to MLP head
    x = self.mlp_head(x[:,0])

    return x
