# UNet-Crack-Detection-Model
This repository contains the implementation of a UNet-based model for crack detection in images. The UNet architecture is widely used for image segmentation tasks due to its effectiveness in capturing both high-level and low-level features. This model is trained to detect cracks in various surfaces, which can be useful in fields like civil engineering and infrastructure maintenance.
# 1. Clone the repository
git clone https://github.com/RishabhGodara/UNet-Crack-Detection-Model.git
cd UNet-Crack-Detection-Model

#2. Training model to your dataset
Although i have trained model , you can train model on your dataset just add you data i.e. images to respective train_images , train_masks , valid_images and valid_masks folder
After that go to src folder ia cd src command and run following command "python3 train.py --epochs "Give total number of epochs" --batch "Give batch size" --lr 0.005"
All the train data will be in outputs folder

#3. Test the model
Upload the images you want to test in test folder
Run the command "python3 inference_image.py --model "Give path to your model" --input "Give path to test folder" --imgsz 512"
You will see output images in inference_results folder in outputs folder
