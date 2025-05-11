# Deon Surfaces Demo

This project demonstrates a **semantic segmentation** application using a pre-trained Segformer model for image segmentation. Built with Django for the backend, this application leverages the Hugging Face `transformers` library to handle image processing and segmentation. It also allows users to interactively place a virtual slab of quartz onto the segmented parts of the image, simulating how a quartz countertop would appear on different segments.

## Features

- **Image Segmentation**: Uses a pre-trained Segformer model to segment images into different categories.
- **Interactive Slab Placement**: Allows the user to select a quartz slab and place it anywhere on the segmented parts of the image (e.g., on a simulated countertop).
- **API Integration**: The backend serves an API endpoint for handling image uploads, segmenting the image, and returning the updated segmented image with the quartz slab overlaid.
- **Overlay Segmentation Map**: The segmented image is overlaid on the original image for visual comparison.
  
## Technologies Used

- **Django**: Web framework for the backend.
- **Django Rest Framework**: For building the API endpoints.
- **Transformers (Hugging Face)**: For using the Segformer model for semantic segmentation.
- **Torch**: PyTorch, used for the model's implementation.
- **OpenCV**: For image processing and manipulation.
- **Matplotlib**: For visualizing segmentation maps.
- **Pillow**: For handling image uploads and manipulation.
- **Dotenv**: For managing environment variables.

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/your-username/deonSurfacesDemo.git
```

2. Navigate to the project directory:
```bash
    cd deonSurfacesDemo
```

3. Install the required dependencies. You can create a virtual environment (optional but recommended):
```bash
    python -m venv venv
    source venv/bin/activate   # For Linux/macOS
    venv\Scripts\activate      # For Windows
```

### Then, install dependencies
```bash
    pip install -r requirements.txt
```

4. Run migrations to set up the database:
```bash
    python manage.py migrate
```

5. Start development server:
```bash
    python manage.py runserver
```
Your app will be running at http://127.0.0.1:8000/.


## Usage

1. **Upload an image** to get a segmented image with the overlay.

2. The model will return a **segmentation map**, which is overlaid onto the original image for visual comparison.

3. Users can **select a slab of quartz** and place it on any part of the segmented image (e.g., on a kitchen countertop or bathroom vanity) to visualize how it would look in that area.
