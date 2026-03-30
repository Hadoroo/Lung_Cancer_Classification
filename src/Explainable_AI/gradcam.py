import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None

        # ambil layer terakhir ResNet
        self.target_layer = model.model.layer4[-1]

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax()

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        grads = self.gradients[0]
        activations = self.activation[0]

        weights = grads.mean(dim=(1,2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224,224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam