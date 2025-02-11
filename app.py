# the file is generated by GenAI, manual work are marked
# package needed: streamlit
# Title of the app
import streamlit as st
from PIL import Image
import torch
import os
import tempfile
from typing import List, Tuple
from torchvision import transforms

loaded_model = torch.jit.load(os.path.join("models","model_scripted.pt"),map_location=torch.device('cpu'))
loaded_model.eval()

# the following function is modified from DL model traing part
# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224)):
    device = "cpu"
    # 1. open image
    img = Image.open(image_path) # /content/drive/MyDrive/AIPI540A1/data/Split/val/Dave

    # 2. create transformation for image

    # apply the same transformations as before.
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=[-0.2, 0.2]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    # 3. Make sure the model is on the target device
    model.to(device)

    # 4. Turn on model evaluation mode and inference mode
    model.eval()

    with torch.inference_mode():
      # 5. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 6. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 7. convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 8. convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    prediction = class_names[target_image_pred_label]

    return prediction, target_image_pred_probs[0,target_image_pred_label].item()


# Add an invisible anchor for scrolling, order is moved manually
st.markdown('<div id="prediction-result-view"></div>', unsafe_allow_html=True)

# Title of the app
st.title("Image Prediction Web App")

# Description
st.write("Upload a JPG image, click Submit Button, and see the prediction result!")

# Placeholder for the prediction result (always visible at the top)
prediction_placeholder = st.empty()
prediction_placeholder.text("Prediction Result: No prediction yet")

# File uploader (accepts only JPG files)
uploaded_file = st.file_uploader("Upload an image", type=["jpg"], accept_multiple_files=False)

# Display the uploaded image with restricted size
if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    # Define the temporary file path
    temp_file_path = os.path.join(temp_dir, "temp.jpg")
    
    # Save the uploaded file to the temporary directory
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    # Open and display the image
    image = Image.open(temp_file_path)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=200)  # Restrict width to 300px, utilize the `use_container_width` parameter instead.Manual changes made here.

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        # manual starts
        class_names = ['Dave', 'Haochen', 'Harshitha', 'Xiaoquan']
        info = {
            'Dave': 'www.linkedin.com/in/ruhan-wang-3b946b175',
            'Haochen': 'www.linkedin.com/in/haochen-harry-li',
            'Harshitha': 'www.linkedin.com/in/harshitha-ras',
            'Xiaoquan': 'www.linkedin.com/in/xiaoquankong'
        }
        # manual ends
        
        # Update the prediction result in the placeholder with plain text
        prediction, pred_probs = pred_and_plot_image(model=loaded_model, image_path=temp_file_path, class_names=class_names, image_size=(224, 224))
        prediction_placeholder.markdown(
            f"Prediction Result: <p style='color:red;'>{prediction}, with probability: {pred_probs*100:.2f} %  <a href='https://{info[prediction]}' target='_blank'>LinkedIn profile</a> </p>",
            unsafe_allow_html=True
        )
        # Scroll to the prediction result using JavaScript, manual changes are made here: "prediction-result-view"
        st.components.v1.html(
            """
            <script>
                var element = window.parent.document.getElementById('prediction-result-view');
                if (element) {
                    element.scrollIntoView({behavior: 'smooth'});
                }
            </script>
            """,
            height=0,
        )
    else:
        st.warning("Please upload an image before clicking Submit.")

