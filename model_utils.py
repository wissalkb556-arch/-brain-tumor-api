import torch, torch.nn as nn, cv2, numpy as np
from torchvision import models, transforms
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
BENIGN_MALIGNANT_MAP = {
    'glioma': 'Malin (à confirmer histologiquement)',
    'meningioma': 'Généralement bénin',
    'pituitary': 'Généralement bénin',
    'notumor': 'N/A',
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path='best_model.pth'):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.gradients, self.activations = model, None, None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()


def predict_and_explain(model, gradcam, image: Image.Image):
    image_rgb = image.convert('RGB')
    tensor = eval_transform(image_rgb).unsqueeze(0).to(device)
    cam, pred_idx, probs = gradcam.generate(tensor)
    cam_resized = cv2.resize(cam, image_rgb.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_np = np.array(image_rgb)
    overlay = (0.5 * original_np + 0.5 * heatmap).astype(np.uint8)
    return {
        'predicted_class': CLASS_NAMES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probabilities': {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        'benign_malignant': BENIGN_MALIGNANT_MAP[CLASS_NAMES[pred_idx]],
        'heatmap_overlay': overlay,
    }
