
from pytorch_alexnet_runner import run_alexnet_pytorch
from pytorch_lenet_runner import run_lenet_pytorch
from pytorch_vgg11_runner import run_vgg_11_pytorch
from AlexNet_TF import run_alexnet_tf_models
from LeNet_TF import run_lenet_tf_models
from VGG11_TF import run_vgg_11_tf_models


if __name__ == "__main__":

    run_alexnet_pytorch()
    run_alexnet_tf_models()

    run_vgg_11_pytorch()
    run_vgg_11_tf_models()
    
    run_lenet_pytorch()
    run_lenet_tf_models() 
