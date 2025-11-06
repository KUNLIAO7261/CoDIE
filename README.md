# CoDIE
Real-Time Image Dehazing via Implicit Neural Encoding: A Lightweight and Efficient Approach
Requirements
python3.10
pytorch==2.3.1
Running the code
python hybrid_main_dehaze.py
The code execution is controlled with the following parameters:

--input_folder defines the name of the folder with input images
--output_folder defines the name of the folder where the output images will be saved
--down_size is the size to which the input image will be downsampled before processing
--epochs defines the number of optimisation steps
--window defines the size of the context window
--L is the "optimally-intense threshold", lower values produce brighter images
The strength of the regularisation terms in the loss functon is defined by the following parameters:

--alpha: fidelity control (default setting: 1)
--beta: illumination smoothness (default setting: 20)
--gamma: exposure control (default setting: 8)
--delta: sparsity level (default setting: 5)
