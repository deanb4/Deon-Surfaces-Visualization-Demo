from django.shortcuts import render
import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from transformers import AutoImageProcessor, SegformerFeatureExtractor, SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from django.conf import settings
from dotenv import load_dotenv
import os
import torchvision.transforms as T
import torch


@api_view(['POST'])
def upload_image(request):
    image = request.FILES.get('image')
    if not image:
        return Response({'error': 'No image uploaded'}, status = 400)
    
    # save image to disk
    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    # img_path = fs.save(image.name,image)
    img_path = fs.path(filename)
    img_url = fs.url(filename)

    # send image to Hugging Face API for segmentation
    result = process_with_ai(img_path)
    if 'error' in result:
        return Response(result, status=500)
    
    return Response(result)

# Currently loading ai model locally not getting from site so dont need .env etc
def process_with_ai(image_path):
    # Get Hugging Face api key & model details
    load_dotenv()
    HF_API_KEY = os.getenv("HF_API_KEY")
    # HF_MODEL_URL = 'https://api-inference.huggingface.co/models/nvidia/segformer-b1-finetuned-ade-512-512'

    if not HF_API_KEY:
        print("Hugging Face API key not loaded!")

    # # nvidia segformer b2 best so far
    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")


    # # load custom trained model
    # model.load_state_dict(torch.load("segformer_final.pth"))

    id2label = model.config.id2label

    headers = {'Authorization': f'Bearer {HF_API_KEY}'} # currently dont need (using model locally)
    with open(image_path, 'rb') as image_file:
        image = Image.open(image_file).convert('RGB') 

        # upscale
        # transform = T.Resize((1024,1024))
        # image = transform(image_original)

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        # Convert logits to a segmentation map (for example)
        segmentation_map = logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # seg_map_mp = np.array(segmentation_map)
        unique_labels = np.unique(segmentation_map)

        # Print segments in human-readable format
        # labels_test = {}
        # for label in unique_labels:
        #     labels_test[label] = id2label.get(label, "Unkown")
        #     print(f"label {label}: {id2label.get(label, 'Unkown')}")

        # # Convert keys to int just once after loading id2label
        id2label = {int(k): v for k, v in id2label.items()}

        label_dict = {int(label): id2label.get(int(label), "Unknown") for label in unique_labels}

        colors = plt.cm.get_cmap('tab20', len(unique_labels))
        # color_dict = {label: colors(i) for i, label in enumerate(unique_labels)}
        color_dict = {label: (np.array(colors(i)[:3]) * 255).astype(np.uint8) for i, label in enumerate(unique_labels)}

        seg_image = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
        for label, color in color_dict.items():
            seg_image[segmentation_map == label] = color # = color[:3]

        # For overlay effect (will need to change as user should not see this)
        # Convert both images to NumPy arrays in BGR for OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        seg_image_bgr = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)

          # Resize segmentation map to match original image
        seg_image_bgr = cv2.resize(seg_image_bgr, (image_cv.shape[1], image_cv.shape[0]))

        alpha = 0.5
        overlay = cv2.addWeighted(image_cv, 1 - alpha, seg_image_bgr, alpha, 0)

        # Sharped Overlay
        sharpen_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        
        sharpened_overlay = cv2.filter2D(overlay, -1, sharpen_kernel)
        output_path = f"media/seg_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, overlay)

        # Testing id2Label
        # try:
        #     id2label = model.config.id2label
        #     print(id2label[0])  # Should print something like 'background' or 'wall'
        # except AttributeError:
        #     print("This model does not have an id2label mapping.")
        # resize if needed
        # if image.shape != seg_image.shape:
        #     seg_image = cv2.resize(seg_image, (image.shape[1], image.shape[0]))

        # uncomment when want to display the matplot lib graph of segmented photo
        # plt.imshow(overlay)
        # plt.axis('off')
        # plt.show()

        # files = {'file': image_file}

        try:
            return {'segmentation_map': segmentation_map.tolist(),
                    "mask_url": f"/{output_path}",
                    "labels": label_dict
                    }
        
        except Exception as e:

            return {"error": str(e)}


def home(request):
    slab_folder = os.path.join(settings.MEDIA_ROOT, 'slabs')
    slabs = os.listdir(slab_folder)
    return render(request, 'home.html', {
        'slabs':slabs,
        'media_url': settings.MEDIA_URL,
        })


# def overlaySegmentImage():
#     original = cv2.imread(r"C:\Users\deanb\OneDrive\dad company project\deonSurfacesDemoImage.jpg")
#     segmented = 