from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision.models import resnet18
import torchvision.models as models
import argparse

import imageio
import cv2
def wrong_predictions(model,test_loader, device):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))

        print(f'Total wrong predictions are {len(wrong_predictions)}')

        plot_misclassified(wrong_predictions, classes)

    return wrong_predictions

def plot_misclassified(wrong_predictions, classes):
    fig = plt.figure(figsize=(10,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*0.5)+0.5

        img = np.transpose(img, (1, 2, 0)) #/ 2 + 0.5
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.set_title(f'\nactual : {classes[target.item()]}\npredicted : {classes[pred.item()]}',fontsize=10)
        ax.imshow(img)


        file_path = os.path.join(os.getcwd(), f'incorrect_image_{i}.jpg')
        imageio.imwrite(file_path, img)
        print(f'Saved image {i} to {file_path}')

        rgb_img[i] = cv2.imread('file_path')



    plt.show()


    