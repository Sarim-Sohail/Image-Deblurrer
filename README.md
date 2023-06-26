This project is a simple image deblurrer, made using MIMO UNET. The addition of a one-image mode to the model was added to further improve user accessbility for image deblurring.  
The new mode can be used using:  

`py -3.8 main.py --model_name "MIMO-UNet" --mode "oneimage" --data_dir "sample" --test_model "MIMO-UNet.pkl" --save_image True`

  
The data_dir should be the directory where the images you want to deblur are
