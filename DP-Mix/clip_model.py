import open_clip
import torch
import torch.nn as nn


class CLIP_model(nn.Module):
    def __init__(self, encoder,NUM_CLASSES,fine_tune_whole_model=False,encoder_output_size=640):
        super(CLIP_model, self).__init__()

        if fine_tune_whole_model==False:
            # Freeze the weights of the encoder
            for param in encoder.parameters():
                param.requires_grad = False


        # Set the encoder as a member variable
        self.encoder = encoder
        self.NUM_CLASSES=NUM_CLASSES

        # Add a linear layer as output
        self.linear_layer = nn.Linear(encoder_output_size, self.NUM_CLASSES)

    def forward(self, input_ids):
        # Pass the input through the encoder
        encoder_output = self.encoder.encode_image(input_ids)

        # Pass the last hidden state through the linear layer
        linear_output = self.linear_layer(encoder_output)

        return linear_output
