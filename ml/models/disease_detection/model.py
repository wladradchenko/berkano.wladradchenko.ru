from timm import create_model
import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaModel

EMBEDDING_DIM = 512


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load the Swin Transformer with features_only=True
        self.swin = create_model("swin_base_patch4_window7_224.ms_in22k", pretrained=True, features_only=True)
        for param in self.swin.parameters():
            param.requires_grad = True

        # Get the feature size of the final stage
        self.swin_output_dim = self.swin.feature_info.channels()[-1]  # Last stage: 1024 channels

        # Define FC layer
        self.fc1 = nn.Linear(self.swin_output_dim * 7 * 7, EMBEDDING_DIM)  # Flattened input size
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        for param in self.fc1.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Extract features from Swin
        swin_features = self.swin(x)[-1]  # Use the last stage feature map (e.g., [B, 1024, 7, 7])

        # Flatten feature map
        swin_features = swin_features.view(swin_features.size(0), -1)  # Shape: (B, 1024*7*7)

        # Pass through FC layer
        output = self.fc1(swin_features)  # Shape: (B, embedding_dim)
        return output


class LVL(nn.Module):
    def __init__(self):
        super(LVL, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = nn.Identity()
        self.t_prime = nn.Parameter(torch.ones([]) * np.log(0.07))
        self.b = nn.Parameter(torch.ones([]) * 0)

    def get_images_features(self, images):
        image_embeddings = self.image_encoder(images)  # (batch_size, EMBEDDING_DIM)
        image_embeddings = nn.functional.normalize(image_embeddings, p=2, dim=-1)
        return image_embeddings

    def get_texts_feature(self, input_ids=None, attention_mask=None):
        """
        Plug
        :param input_ids: Tensor of shape (batch_size, seq_length)
        :param attention_mask: Tensor of shape (batch_size, seq_length)
        :return:
        """
        return None

    def forward(self, images, input_ids=None, attention_mask=None):
        """
        Args:
            images: Tensor of shape (batch_size, 3, 224, 224)
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Tensor of shape (batch_size, seq_length)

        Returns:
            Image and text embeddings normalized for similarity calculation
        """

        image_embeddings = self.get_images_features(images)
        return image_embeddings
