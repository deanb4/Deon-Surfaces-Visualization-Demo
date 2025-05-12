const canvas = new fabric.Canvas('canvas');
let selectedSlab = null;
let draggedSlabSrc = null;
let segmentationMap = [];
let height = 0;
let width = 0;

document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const dropdown = document.getElementById('room-select');
    let dropdownFile = null;

    if (!fileInput.files.length && !dropdown) return alert("Please select an image");

    const file = fileInput.files[0];
    const dropdownValue = dropdown.value;
    console.log(dropdownValue)

    if (!file && !dropdownValue){
        return alert("Please select an image from dropdown or upload your own."); 
    } 

    const formData = new FormData();

    if (file) {
        formData.append('image', fileInput.files[0]);
        fileInput.value =""; 
    } else if (dropdownValue) {
        console.log("dropdown value selected")
        try {
            const response = await fetch(dropdownValue);
            const blob = await response.blob()
            const filename = dropdownValue.split('/').pop();
            dropdownFile = new File([blob], filename, { type: blob.type });
            formData.append('image', dropdownFile);
        } catch (err) {
            return alert("Failed to load image from dropdown menu: " + err.message);
        }
    }
    //     const response = await fetch('http://127.0.0.1:8000/upload/', {
    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData 
        });

        const data = await response.json();

        if (data.error) {
            return alert('AI processing failed: ' + data.error);
        }

        // If all good, render the canvas using fabric
        // renderCanvasImages(fileInput.files[0], data.mask_url);
        if (file) {
            renderCanvasImages(file, data.mask_url);
            console.log("rendering")
        } else if (dropdownFile) {
            renderCanvasImages(dropdownFile, data.mask_url);
            console.log("rendering")
        }
        fetchSegmentationMap(data.segmentation_map)

    } catch (err) {
        alert("Upload failed " + err.message);
    }

});

function renderCanvasImages(file, maskUrl) {
    const reader = new FileReader();
    reader.onload = () => {
        // Reset
        canvas.clear();
        canvas.setBackgroundImage(null, canvas.renderAll.bind(canvas));

        fabric.Image.fromURL(reader.result, function(img) {
            const originalWidth = img.width;
            const originalHeight = img.height;
            height = originalHeight;
            width = originalWidth;
            console.log("Base Image: ", img.width, img.height);
            // Max display size
            const maxCanvasWidth = 800;
            const scale = Math.min(maxCanvasWidth / originalWidth, 1); // don't upscale

            const canvasWidth = originalWidth * scale;
            const canvasHeight = originalHeight * scale;

            canvas.setWidth(canvasWidth);
            canvas.setHeight(canvasHeight);

            canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
                scaleX: scale,
                scaleY: scale
            });

            // add segmentation mask
            fabric.Image.fromURL(maskUrl, function(maskImg) {
                maskImg.set({
                    left: 0,
                    top: 0,
                    scaleX: scale,
                    scaleY: scale,
                    selectable: false,
                    opacity: 0.2 // change to 0 so mask does not show up on front end
                });
                canvas.add(maskImg);
                console.log("Segmentation Mask Image: ", maskImg.width, maskImg.height);
            });
            console.log('Segmented image URL: ', maskUrl)
            // addSlabImage();
        });
        
        

    };
    reader.readAsDataURL(file);
}


function fetchSegmentationMap(segmentation_map) {
    segmentationMap = JSON.parse(JSON.stringify(segmentation_map));
    console.log("Segmentation Map Dimensions: ", segmentationMap[0].length, segmentationMap.length);
}

// Unused (click method)
function addSlabImage() {
    if (selectedSlab){
        fabric.Image.fromURL(`/media/slabs/${selectedSlab}`, function(slabImg) {
            slabImg.set({
                left: 100,
                top: 100,
                scaleX: 0.3,
                scaleY: 0.3,
                hasControls: true,
                hasBorders: true,
                selectable: true
            });

            // Add drag and drop functionalit
            slabImg.on('mouse:down', function (e) {
                // Get mouse position on canvas
                const pointer = canvas.getPointer(e.e);
                slabImg.set({
                    left: pointer.x - (slabImg.width * slabImg.scaleX) / 2,
                    top: pointer.y - (slabImg.height & slabImg.scaleY) /2
                });
                canvas.renderAll();
            });
            canvas.add(slabImg)
        });
    }
}


// Handle the slab selection
// document.querySelectorAll('.slab-thumbnail').forEach((thumbnail) => {
//     thumbnail.addEventListener('click', function () {
//         selectedSlab = this.dataset.slab;
//         alert('Slab selected: ' + selectedSlab);
//     });
// });


document.querySelectorAll('.slab-img').forEach((thumbnail) => {
    thumbnail.addEventListener('dragstart', (e) => {
        draggedSlabSrc = e.target.dataset.src;
    });
});

// Handle drop on Fabric canbas container
const canvasWrapper = document.getElementById('canvas-wrapper');
canvasWrapper.addEventListener('dragover', (e) => {
    e.preventDefault(); // allow drop
});

canvasWrapper.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!draggedSlabSrc) return;
    
    const canvasRect = canvasWrapper.getBoundingClientRect();
    const offsetX = e.clientX - canvasRect.left;
    const offsetY = e.clientY - canvasRect.top;

    // Convert drop position to segmentation coordinates
    const segmentationMapWidth = segmentationMap[0].length;
    const segmentationMapHeight = segmentationMap.length;

    // const scaleX = (canvas.width / width) * segmentationMapWidth;
    // const scaleY = (canvas.height / height) * segmentationMapHeight;

    // const segmentationX = Math.floor(offsetX / scaleX);
    // const segmentationY = Math.floor(offsetY / scaleY);
    // ADDED
    const imageToCanvasScale = canvas.width / width;

    const originalX = offsetX / imageToCanvasScale;
    const originalY = offsetY / imageToCanvasScale;

    const segmentationX = Math.floor((originalX / width) * segmentationMap[0].length);
    const segmentationY = Math.floor((originalY / height) * segmentationMap.length);

    // Check the segmentation value at the drop position
    // const segmentationValue = segmentationMap[segmentationY] ? segmentationMap[segmentationY][segmentationX] : null;
    const segmentationValue = segmentationMap[segmentationY]?.[segmentationX];


    console.log("Segmentation coordinates:", segmentationX, segmentationY);
    console.log("Segmentation value at drop:", segmentationValue);

    if (segmentationValue === null || segmentationValue === 0) {
        alert("Slab can't be places in this region.");
        return;
    }

    // Find segmentation mask layer
    const mask = canvas.getObjects().find(obj => obj.type === 'image' && obj.selectable === false && obj.opacity <= 0.2);
    if (!mask) {
        alert("Segmentation mask not found");
        return;
    }
    
    // const depthValue = fetchDepthValue(segmentationX, segmentationY);
    
    // // Calculate scale factor based on depth
    // const scaleFactor = calculateScaleFactor(depthValue);
    
    // // Load the slab image and apply texture or other transformations
    // fabric.util.loadImage(draggedSlabSrc, (img) => {
        //     // Create the slab object with the adjusted scale based on depth
    //     const pattern = new fabric.Image(img, {
        //         left: offsetX - (img.width * scaleFactor) / 2, // Adjust position based on scale
        //         top: offsetY - (img.height * scaleFactor) / 2, // Adjust position based on scale
        //         scaleX: scaleFactor,
        //         scaleY: scaleFactor,
    //         hasControls: true,
    //         hasBorders: true,
    //         selectable: true,
    //     });
    
    fabric.util.loadImage(draggedSlabSrc, (img) => {
        const pattern = new fabric.Pattern({
            source: img,
            repeat: 'repeat'
        });
    
        // Create a large rectangle and fill it with the slab pattern
        const texturedRect = new fabric.Rect({
            left: 0,
            top: 0,
            width: canvas.width,
            height: canvas.height,
            fill: pattern,
            selectable: true,
            hasControls: true,
            hasBorders: true,
            originX: 'left',
            originY: 'top',
            customType: 'slab'
        });
    
        // Apply the segmentation mask as the clip path
        const maskDataUrl = createSegmentationMask(segmentationValue);
        fabric.Image.fromURL(maskDataUrl, (clipImg) => {
        clipImg.set({
            scaleX: canvas.width / clipImg.width,
            scaleY: canvas.height / clipImg.height,
            originX: 'left',
            originY: 'top',
            left: 0,
            top: 0
        });

        texturedRect.clipPath = clipImg;
        clipImg.absolutePositioned = true;
        canvas.add(texturedRect);
        canvas.setActiveObject(texturedRect);
        canvas.renderAll();
        });
    });
    
    draggedSlabSrc = null;
});


// // testing
function createSegmentationMask(segmentationValue) {
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = segmentationMap[0].length;
    maskCanvas.height = segmentationMap.length;

    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.createImageData(maskCanvas.width, maskCanvas.height);
    const data = imageData.data;

    for (let y = 0; y < segmentationMap.length; y++) {
        for (let x = 0; x < segmentationMap[0].length; x++) {
            const idx = (y * maskCanvas.width + x) * 4;
            const val = segmentationMap[y][x];
            if (val === segmentationValue) {  // You can customize this if you want a specific label
                data[idx] = 255;     // R
                data[idx + 1] = 255; // G
                data[idx + 2] = 255; // B
                data[idx + 3] = 255; // A (fully visible)
            } else {
                data[idx + 3] = 0;   // transparent
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);
    return maskCanvas.toDataURL();
}



function applyTextureToSegment(segmentationMap, segmentValue, textureImage, baseCanvas) {
    const width = segmentationMap[0].length;
    const height = segmentationMap.length;

    const ctx = baseCanvas.getContext('2d');

    // Create an offscreen canvas to draw the texture
    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = width;
    textureCanvas.height = height;
    const textureCtx = textureCanvas.getContext('2d');

    // Tile the texture to fit the entire canvas
    const pattern = textureCtx.createPattern(textureImage, 'repeat');
    textureCtx.fillStyle = pattern;
    textureCtx.fillRect(0, 0, width, height);

    // Get image data from the tiled texture
    const textureData = textureCtx.getImageData(0, 0, width, height).data;

    // Create an image object for the base canvas
    const textureFabricImage = new fabric.Image(textureCanvas);

    // Get base image data to modify
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            if (segmentationMap[y][x] === segmentValue) {
                // Replace base image pixel with texture pixel
                data[idx]     = textureData[idx];     // R
                data[idx + 1] = textureData[idx + 1]; // G
                data[idx + 2] = textureData[idx + 2]; // B
                data[idx + 3] = 255;                  // A
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);
     // Now add the texture image to the canvas (using fabric.Image)
    // Apply edge detection and sharpening
    baseCanvas.add(textureFabricImage);
}


function loadTextureAndApply(segmentValue) {
    const textureImage = new Image();
    textureImage.src = '/../media/slabs/vulcanica_slab.png';  // Replace with your actual texture path

    textureImage.onload = () => {
        // Create an offscreen canvas the same size as the segmentation map
        const baseCanvas = new fabric.Canvas('canvas');;
        baseCanvas.width = segmentationMap[0].length;
        baseCanvas.height = segmentationMap.length;

        applyTextureToSegment(segmentationMap, segmentValue, textureImage, baseCanvas);

        // Optional: show the result on screen
        document.body.appendChild(baseCanvas.getElement());
    };
}

// Function to fetch the depth value for a specific position in the segmentation map
async function fetchDepthValue(x, y) {
    // Prepare form data to send to the backend
    const formData = new FormData();
    formData.append("x", x);
    formData.append("y", y);

    // Send POST request to backend
    const response = await fetch("/render/", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Failed to fetch depth data');
    }

    // Assuming the backend returns the depth value in the response body
    const data = await response.json();

    // Extract and return the depth value from the response
    return data.depthValue; // Adjust based on your backend's response format
}

// Function to calculate scale factor based on the depth value
function calculateScaleFactor(depth) {
    // Simple example: inverse depth scaling (closer objects appear larger)
    const maxDepth = 100; // Max depth value (adjust as needed)
    const minScale = 0.2; // Minimum scale
    const maxScale = 1; // Maximum scale

    // Scale inversely with depth (closer slabs are bigger)
    const scale = (maxDepth - depth) / maxDepth;
    return minScale + scale * (maxScale - minScale);
}


document.getElementById('resetSlabsBtn').addEventListener('click', () => {
    const slabsToRemove = canvas.getObjects().filter(obj => obj.customType == 'slab');
    slabsToRemove.forEach(slab => canvas.remove(slab));
    canvas.renderAll();
});
// function changeRoomBackground() {
//     var select = document.getElementById("room-select");
//     var roomImageUrl = select.value;

//     if (roomImageUrl) {
//         fabric.Image.fromUrl(roomImageUrl, function(img) {
//             // Remove any previous background image
//             canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas), {
//                 scaleX: canvas.width / img.width,
//                 scaleY: canvas.height / img.height
//             });
//         });
//     }

// }























    // took out of get element by id  function (first one) 
    // // Display the uploaded image on the canvas
    // const reader = new FileReader();
    // reader.onload = () => {
    //     const img = new Image();
    //     img.onload = () => {
    //         const canvas = document.getElementById('canvas');
    //         const ctx = canvas.getContext('2d');
    //         canvas.width = img.width;
    //         canvas.height = img.height;
    //         ctx.drawImage(img,0,0);
            
    //         // Overlay the segmentation mask
    //         const maskImage = new Image();
    //         maskImage.src = data.mask_url;
    //         maskImage.onload = () => {
    //             ctx.drawImage(maskImage, 0, 0);
    //         };
    //     };
    //     img.src = reader.result;
    // };
    // reader.readAsDataURL(fileInput.files[0]);



    // canvasWrapper.addEventListener('drop', async (e) => {
    //     e.preventDefault();
    //     if (!draggedSlabSrc) return;
    
    //     // Load the slab image first
    //     const slabImg = new Image();
    //     slabImg.crossOrigin = "Anonymous";
        
    //     try {
    //         // Load the slab image
    //         await new Promise((resolve, reject) => {
    //             slabImg.onload = resolve;
    //             slabImg.onerror = reject;
    //             slabImg.src = draggedSlabSrc;
    //         });
            
    //         // Create a temporary canvas to draw the result
    //         const tempCanvas = document.createElement('canvas');
    //         tempCanvas.width = canvas.width;
    //         tempCanvas.height = canvas.height;
    //         const tempCtx = tempCanvas.getContext('2d');
            
    //         // First draw the current canvas state
    //         tempCtx.drawImage(canvas.getElement(), 0, 0);
            
    //         // Calculate scaling from segmentation map to canvas
    //         const scaleX = canvas.width / segmentationMap[0].length;
    //         const scaleY = canvas.height / segmentationMap.length;
            
    //         // Loop through segmentation map and draw slab where value is 3
    //         for (let y = 0; y < segmentationMap.length; y++) {
    //             for (let x = 0; x < segmentationMap[0].length; x++) {
    //                 if (segmentationMap[y][x] === 3) { // 3 is floor
    //                     // Draw a small piece of the slab image at this position
    //                     tempCtx.drawImage(
    //                         slabImg,
    //                         0, 0, slabImg.width, slabImg.height,
    //                         x * scaleX, y * scaleY, scaleX, scaleY
    //                     );
    //                 }
    //             }
    //         }
            
    //         // Create fabric image from the result
    //         fabric.Image.fromURL(tempCanvas.toDataURL(), (img) => {
    //             // Clear canvas and add this new image
    //             canvas.clear();
    //             canvas.add(img);
    //             canvas.renderAll();
    //         });
            
    //     } catch (error) {
    //         console.error("Error loading slab image:", error);
    //     }
        
    //     draggedSlabSrc = null;
    // });

