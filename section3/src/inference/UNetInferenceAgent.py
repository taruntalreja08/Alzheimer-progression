"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        slices.append(volume[0:1])
        arr = np.zeros(volume.shape, dtype=np.float32) 
        
        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                label = label.split(" ")
                mask = np.zeros(volume.shape[1] * volume.shape[2], dtype=np.uint8)

                posit = map(int, label[0::2])
                leng = map(int, label[1::2])

                for p, l in zip(posit, leng):
                    mask[p:(p+l)] = 1
                arr[:, :, idx] = mask.reshape(volume.shape[1], volume.shape[1], order='F')

        slices = slices.asarray()
        slices.reshape((-1, slices.shape[0], slices.shape[1]))
        return slices
