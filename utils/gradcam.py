from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision.models import resnet18
import torchvision.models as models
import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig = plt.figure(figsize=(20, 20))

for i in range(1,10):
    img = mpimg.imread(f'/content/incorrect_image_{i}.jpg')
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda='cuda')
    #targets = [ClassifierOutputTarget(281)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    image = visualization
    fig.add_subplot(5, 4, i)
    plt.imshow(image)
    plt.title(classes[i])

plt.show()
