import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms():
    
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    # training same architecture
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),   # return logit
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_threshold(path="models/threshold.json", default=0.5):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["threshold"]
    return default


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(x)            # logit shape (1, 1)
        output[0, 0].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)

        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def remove(self):
        
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
    img_np = np.array(original_image_pil.resize((IMG_SIZE, IMG_SIZE)))
    h, w, _ = img_np.shape
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)





#################
# import json
# import os
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import torch.nn.functional as F
# import cv2
# import numpy as np

# IMG_SIZE = 224
# MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# def get_transforms():
#     # لازم يطابق eval_tf في التدريب بالظبط
#     return transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=MEAN, std=STD),
#     ])


# def load_model(model_path):
#     model = models.efficientnet_b0(weights=None)
#     in_features = model.classifier[1].in_features
#     # نفس architecture التدريب بالظبط
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3),
#         nn.Linear(in_features, 1),   # بيرجع logit
#     )
#     state = torch.load(model_path, map_location="cpu")
#     model.load_state_dict(state)
#     model.eval()
#     return model


# def load_threshold(path="models/threshold.json", default=0.5):
#     if os.path.exists(path):
#         with open(path) as f:
#             return json.load(f)["threshold"]
#     return default


# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.activations = None
#         self.gradients = None
#         self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
#         self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

#     def _save_activation(self, module, inp, out):
#         self.activations = out.detach()

#     def _save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def __call__(self, x):
#         self.model.eval()
#         self.model.zero_grad()
#         output = self.model(x)            # logit shape (1, 1)
#         output[0, 0].backward()

#         gradients = self.gradients[0]
#         activations = self.activations[0]
#         weights = gradients.mean(dim=(1, 2), keepdim=True)

#         cam = (weights * activations).sum(dim=0)
#         cam = F.relu(cam)
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
#         return cam.cpu().numpy()

#     def remove(self):
#         # مهم: عشان الـ hooks متتراكمش على الموديل مع كل طلب
#         self._fwd_handle.remove()
#         self._bwd_handle.remove()


# def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
#     img_np = np.array(original_image_pil.resize((IMG_SIZE, IMG_SIZE)))
#     h, w, _ = img_np.shape
#     cam_resized = cv2.resize(cam, (w, h))

#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     return cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)







#####################
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import torch.nn.functional as F
# import cv2
# import numpy as np

# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])

# def load_model(model_path):
#     model = models.efficientnet_b0(weights=None)
#     num_ftrs = model.classifier[1].in_features
    
#     # تعديل الطبقة الأخيرة لتخرج قيمة واحدة (1) تطابق ملف الـ Checkpoint
#     model.classifier[1] = nn.Linear(num_ftrs, 1)
    
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # كلاس Grad-CAM (محدث ليتوافق مع Output = 1)
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def __call__(self, x, class_idx=0):
#         self.model.eval()
#         output = self.model(x)
            
#         self.model.zero_grad()
#         class_score = output[0, 0]
#         class_score.backward()

#         gradients = self.gradients[0]
#         activations = self.activations[0]
#         weights = gradients.mean(dim=(1, 2), keepdim=True)
        
#         cam = (weights * activations).sum(dim=0)
#         cam = F.relu(cam)
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
        
#         return cam.cpu().numpy()

# def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
#     img_np = np.array(original_image_pil.resize((224, 224)))
#     h, w, _ = img_np.shape
#     cam_resized = cv2.resize(cam, (w, h))
    
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
#     superimposed_img = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
#     return superimposed_img


#################
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import torch.nn.functional as F
# import cv2
# import numpy as np

# # 1. تجهيز الـ Transforms للصورة
# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])

# # 2. تحميل الموديل (EfficientNet-B0)
# def load_model(model_path):
#     # استخدام EfficientNet مطابقة للتي تم التدريب عليها
#     model = models.efficientnet_b0(weights=None)
    
#     # تعديل طبقة التصنيف الأخيرة لتناسب فئتين (NORMAL و PNEUMONIA)
#     num_ftrs = model.classifier[1].in_features
#     model.classifier[1] = nn.Linear(num_ftrs, 2)
    
#     # تحميل الأوزان المحفوظة
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # 3. كلاس الـ Grad-CAM
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def __call__(self, x, class_idx=None):
#         self.model.eval()
#         output = self.model(x)
#         if class_idx is None:
#             class_idx = output.argmax(dim=1).item()
            
#         self.model.zero_grad()
#         class_score = output[0, class_idx]
#         class_score.backward()

#         gradients = self.gradients[0]
#         activations = self.activations[0]
#         weights = gradients.mean(dim=(1, 2), keepdim=True)
        
#         cam = (weights * activations).sum(dim=0)
#         cam = F.relu(cam)
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
        
#         return cam.cpu().numpy(), class_idx

# # 4. دمج الخريطة الحرارية بالصورة الأصلية
# def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
#     img_np = np.array(original_image_pil.resize((224, 224)))
#     h, w, _ = img_np.shape
#     cam_resized = cv2.resize(cam, (w, h))
    
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
#     superimposed_img = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
#     return superimposed_img



####################
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import torch.nn.functional as F
# import cv2
# import numpy as np

# # 1. تجهيز الـ Transforms للصورة
# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])

# # 2. تحميل الموديل (تأكد إن الهيكل بيطابق الموديل اللي درّبته، سواء DenseNet أو EfficientNet)
# def load_model(model_path):
#     # لو كنت مدرب DenseNet121:
#     model = models.densenet121(weights=None)
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Sequential(nn.Linear(num_ftrs, 2))
    
#     # تحميل الأوزان
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # 3. كلاس الـ Grad-CAM
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def __call__(self, x, class_idx=None):
#         self.model.eval()
#         output = self.model(x)
#         if class_idx is None:
#             class_idx = output.argmax(dim=1).item()
            
#         self.model.zero_grad()
#         class_score = output[0, class_idx]
#         class_score.backward()

#         gradients = self.gradients[0]
#         activations = self.activations[0]
#         weights = gradients.mean(dim=(1, 2), keepdim=True)
        
#         cam = (weights * activations).sum(dim=0)
#         cam = F.relu(cam)
#         cam = cam - cam.min()
#         cam = cam / (cam.max() + 1e-8)
        
#         return cam.cpu().numpy(), class_idx

# # 4. دمج الخريطة الحرارية بالصورة الأصلية
# def generate_heatmap_image(original_image_pil, cam, alpha=0.4):
#     img_np = np.array(original_image_pil.resize((224, 224)))
#     h, w, _ = img_np.shape
#     cam_resized = cv2.resize(cam, (w, h))
    
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
#     superimposed_img = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
#     return superimposed_img



###################
# import torch
# import torch.nn as nn
# from torchvision import models
# import numpy as np

# class EfficientNetPneumonia(nn.Module):
#     def __init__(self):
#         super(EfficientNetPneumonia, self).__init__()
#         self.model = models.efficientnet_b0(weights=None)
#         num_ftrs = self.model.classifier[1].in_features
#         self.model.classifier = nn.Sequential(
#             nn.Dropout(p=0.3, inplace=True),
#             nn.Linear(num_ftrs, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)

# def generate_gradcam_efficientnet(model, image_tensor):
#     model.eval()
#     gradients = []
#     activations = []

#     # ربط الهوك بأخر طبقة تفاعلية في EfficientNet features لعمل Grad-CAM
#     target_layer = model.model.features[-1]

#     def save_gradient(grad):
#         gradients.append(grad)

#     def forward_hook(module, input, output):
#         activations.append(output)
#         output.register_hook(save_gradient)

#     handle = target_layer.register_forward_hook(forward_hook)
    
#     output = model(image_tensor)
#     model.zero_grad()
#     output.backward()

#     handle.remove()

#     pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
#     activation = activations[0][0]

#     for i in range(activation.shape[0]):
#         activation[i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(activation, dim=0).squeeze()
#     heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
#     heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

#     return heatmap