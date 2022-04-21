
from pytorch_alexnet_runner import run_alexnet_pytorch, run_decomp_alexnet_pytorch
from pytorch_lenet_runner import run_lenet_pytorch, run_decomp_lenet_pytorch
from pytorch_vgg11_runner import run_vgg_11_pytorch, run_decomp_vgg11_pytorch
from AlexNet_TF import run_alexnet_tf_models, run_alexnet_tf_decomposed
from LeNet_TF import run_lenet_tf_models, run_lenet_tf_decomposed
from VGG11_TF import run_vgg_11_tf_models, run_vgg11_tf_decomposed
from constants import Decomposition


if __name__ == "__main__":

    run_lenet_pytorch()
    run_decomp_lenet_pytorch(Decomposition.CP)
    run_decomp_lenet_pytorch(Decomposition.Tucker)

    run_alexnet_pytorch()
    run_decomp_alexnet_pytorch(Decomposition.CP)
    run_decomp_alexnet_pytorch(Decomposition.Tucker)

    #run_vgg_11_pytorch()
    #run_decomp_vgg11_pytorch(Decomposition.CP)
    #run_decomp_vgg11_pytorch(Decomposition.Tucker)


    run_lenet_tf_models()
    #run_lenet_tf_decomposed(Decomposition.CP)
    run_lenet_tf_decomposed(Decomposition.Tucker)

    run_alexnet_tf_models()
    #run_alexnet_tf_decomposed(Decomposition.CP)
    #run_alexnet_tf_decomposed(Decomposition.Tucker)

    run_vgg_11_tf_models()
    #run_vgg11_tf_decomposed(Decomposition.CP)
    #run_vgg11_tf_decomposed(Decomposition.Tucker)