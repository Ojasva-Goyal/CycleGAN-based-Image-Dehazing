# Enhancing Image Clarity through CycleGAN-based Dehazing

This project implements a CycleGAN-based method for image dehazing. It uses a paired dataset containing hazy images and their corresponding ground truth clear images to train the CycleGAN model. The trained model can then be used to dehaze new images.


### Prerequisites

- Python (>=3.6)
- PyTorch
- torchvision
- PIL

### Script Parameters
The testing script accepts several parameters to control its execution:

--test_data_directory: Path to the directory containing test hazy images.

--output_directory: Path for the directory where dehazed images will be saved.

--model_path: Path to the pre-trained CycleGAN model file.
It's in the same folder as in testing script.

### Running the testing script

```bash
python3 image_dehazing_testing_script.py
```

### Output Format
The script generates dehazed images saved in the specified output directory. Each dehazed image filename corresponds to the input hazy image filename.

# Troubleshooting
Permission Errors: If you encounter permission errors when writing the output images, ensure the script has write access to the output directory or try running the script with elevated permissions.

Model Not Found: Verify the model file is placed correctly and the path provided to `--model_path` parameter is accurate.

# Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

# Contact
Created by `Ojasva Goyal` - feel free to contact me at ojasvagoyal9@gmail.com for any questions or feedback.


