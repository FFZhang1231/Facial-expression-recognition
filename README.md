# Joint Pose and Expression Modeling for Facial Expression Recognition

##Pre-requisites
 (1) Python 2.7.
 
 (2) Scipy.
 
 (3) TensorFlos (r0.12) .
 
 *Please note that you will get errors if running with TensorFlow r1.0 or higher version because the definition of input arguments of some function have changed, e.g., 'tf.concat', 'tf.nn.sigmoid_cross_entropy_with logits', and 'tf.nn.softmax_cross_entropy_with_logits'*

 ##Datasets
 (1) You may use any dataset with labels of expression and pose. In our experiments, we use Multi-PIE, BU-3DFE, and SFEW. 
 
 (2) It is better to detect the face before you train the model. In this paper, we use a lib face detection algorithm (https://github.com/ShiqiYu/libfacedetection)

 ##Training
 ```
 $ python mainexpression.py
 ```

 During training, two new folders named 'PFER', and 'result', and one text named 'testname.txt' will be created. 

 'PFER': including four sub-folders: 'checkpoint', 'test', 'samples', and 'summary'.

 (1) 'checkpoint' saves the model
 
 (2) 'test' saves the testing results at each epoch (generated facial images with different expressions based on the input faces).
 
 (3) 'samples' saves the reconstructed facial images at each epoch on the sample images. 
 
 (4) 'summary' saves the batch-wise losses and intermediate outputs. To visualize the summary.
 
 *You can use tensorboard to visualize the summary.*
 
```
 $ cd PFER/summary
 $ tensorboard --logidr . 
```

 *After training, you can check the folders 'samples' and 'test' to visualize the reconstruction and testing performance, respectively.*

 'result': including three kinds of '.txt' files: '*epoch.txt', '*index.txt', and '*test.txt', where * means the number of epoch.
 
 (1) '*epoch.txt' saves the training results at each epoch on the training data. 
 
 (2) '*index.txt' saves the classified labels for each test image.
 
 (3) '*test.txt' saves the number of the test images that classified right.
 

 'testname.txt': including the name of all the test images, which you can use to calculate the facial expression recognition accuracy over each pose and expression. 

##Testing

```
$ python mainexpression.py --is_train False --testdir path/to/your/test/subset/
```

Then, it is supposed to print out the following message.

****
Building graph ...

Testing Mode

Loading pre-trained model ...
SUCCESS

Generate images with different expressions and poses

Done! Results are saved as PFER/test/test_as_xxx.png
***

## Files
(1) 'Facial_expression_train.py' is a class that builds and initializes the model, and implements training and testing related stuff.

(2) 'ops.py' consists of functions required in 'Facial_expression_train.py' to implement options of convolution, deconvolution, fully connection, max_pool, avg_pool, leaky Relu, and so on.

(3) 'mainexpression.py' demonstrates 'Facial_expression_train.py'.
