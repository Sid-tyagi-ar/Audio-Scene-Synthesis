import torch
from torch import nn
from einops import rearrange


class PixelSNAILTransformer(nn.Module):
    def __init__(self, shape, n_class, d_model, nhead, num_layers):
        super().__init__()
        height, width = shape

        # Adjust in_conv to match input channels (1) and output 512 channels
        self.in_conv = nn.Conv2d(1, 512, kernel_size=3, padding=1)

        self.pos_embedding = nn.Embedding(height * width, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers
        )

        # Adjust out_conv to accept 512 input channels
        self.out_conv = nn.Conv2d(512, n_class, kernel_size=1)
    def generate_causal_mask(self, height, width):  
        mask = torch.triu(torch.ones((height, width)))
        return mask.type(torch.bool) 
    def forward(self, input):
        batch_size, height, width = input.shape
        input = input.float()
        x = self.in_conv(input)  # Process class probabilities with convolution
        x = rearrange(x, 'b h w -> b (h w)')  # Reshape for Transformer (batch, seq_len, channels)

        pos = torch.arange(0, height * width, device=x.device).unsqueeze(1)  # Positional encoding
        pos = self.pos_embedding(pos)

        mask = self.generate_causal_mask(height, width)  # Create causal mask
        memory = x
        x = self.transformer(x, tgt_mask=mask , memory = memory)  # Pass through causal transformer layers

        x = rearrange(x, 'b (h w) c -> b c h w')  # Reshape back to image format

        out = self.out_conv(x)
        return out

# Here's the modified main function to use PixelSNAILTransformer:

# Python
if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from datasets import LMDBDataset
  from torch import nn

  device = 'cuda'

  dataset = LMDBDataset('code/')
  loader = DataLoader(
      dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True
  )

  # Instantiate the transformer model with appropriate hyperparameters
  model = PixelSNAILTransformer(
      shape=[20, 86],  # Shape of the input image
      n_class=512,  # Number of class labels
      d_model=128,  # Transformer embedding dimension
      nhead=4,  # Number of Transformer heads
      num_layers=2  # Number of Transformer layers
  )

  model = nn.DataParallel(model)
  model = model.to(device)

  for i, (bottom, class_id, salience, file_name) in enumerate(loader):
    class_id = (
        torch.FloatTensor(list(map(eval, list(class_id)))).long().unsqueeze(1)
    )
    salience = torch.FloatTensor(list(map(eval, list(salience)))).unsqueeze(1)

    # Pass only the "bottom" input to the transformer model
    out, _ = model(bottom)  # No label_condition or salience_condition needed

    if i == 5:
      print(class_id, salience, file_name)
      break