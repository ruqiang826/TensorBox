<img src=http://russellsstewart.com/s/tensorbox/tensorbox_output.jpg></img>

### Tensorflow == 0.11 supported by current version (Please report if you use >0.11 and get errors).

TensorBox is a simple framework for training neural networks to detect objects in images. 
Training requires a json file (e.g. [here](http://russellsstewart.com/s/tensorbox/test_boxes.json))
containing a list of images and the bounding boxes in each image.
The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. We additionally provide an implementation of the 
[ReInspect](https://github.com/Russell91/ReInspect/)
algorithm, reproducing state-of-the-art detection results on the highly occluded TUD crossing and brainwash datasets.

## OverFeat Installation & Training
First, [install TensorFlow from source or pip](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation) (NB: source installs currently break threading on 0.11)
    
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    $ cd /path/to/tensorbox/utils && make && cd ..
    $ python train.py --hypes hypes/overfeat_rezoom.json --gpu 0 --logdir output

Note that running on your own dataset should only require modifying the `hypes/overfeat_rezoom.json` file. 
When finished training, you can use code from the provided 
[ipython notebook](https://github.com/Russell91/tensorbox/blob/master/evaluate.ipynb)
to get results on your test set.

## ReInspect Installation & Training

ReInspect, [initially implemented](https://github.com/Russell91/ReInspect) in Caffe,
is a neural network extension to Overfeat-GoogLeNet in Tensorflow.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details or the <a href="https://www.youtube.com/watch?v=QeWl0h3kQ24" target="_blank">video</a> for a demonstration.

    # REQUIRES TENSORFLOW VERSION >= 0.8
    $ git clone http://github.com/russell91/tensorbox
    $ cd tensorbox
    $ ./download_data.sh
    
    $ # Download the cudnn version used by your tensorflow verion and 
    $ # put the libcudnn*.so files on your LD_LIBRARY_PATH e.g.
    $ cp /path/to/appropriate/cudnn/lib64/* /usr/local/cuda/lib64

    $ cd /path/to/tensorbox/utils && make && make hungarian && cd ..
    $ python train.py --hypes hypes/lstm_rezoom.json --gpu 0 --logdir output

## Tensorboard

You can visualize the progress of your experiments during training using Tensorboard.

    $ cd /path/to/tensorbox
    $ tensorboard --logdir output
    $ # (optional, start an ssh tunnel if not experimenting locally)
    $ ssh myserver -N -L localhost:6006:localhost:6006
    $ # open localhost:6006 in your browser
    
For example, the following is a screenshot of a Tensorboard comparing two different experiments with learning rate decays that kick in at different points. The learning rate drops in half at 60k iterations for the green experiment and 300k iterations for red experiment.
    
<img src=http://russellsstewart.com/s/tensorbox/tensorboard_loss.png></img>

1. this line cause a error:
  from utils import train_utils, googlenet_load

  put it  just before "import tensorflow.contrib.slim as slim", will be OK. I havn't figure out why.
2.  run evaluate.ipynb
  pip install jupyter
  jupyter nbconvert evaluate.ipynb --to python --output evaluate.ipynb # be careful it will replace evaluate.py without --output
  delete the line "get_ipython()..." if you want to run with "python"
  python evaluate.ipynb.py

  or just run :(need jupyter installed)
  ipython notebook evaluate.ipynb
  
3. save or show figure
  save figure : 
  add "plt.savefig("pred_%d" % i)" after "plt.imshow(new_img)"

  show figure:
  add "plt.show()" after "plt.imshow(new_img)"
  

关于train：
1. 生成训练数据在 train_utils.load_data_gen -> load_idl_tf -> al.parse && annotation_to_h5 .
2. 主要的逻辑在al.parse && annotation_to_h5. 这里根据配置文件的    "image_width" "image_height" "grid_height" "grid_width" "batch_size" "region_size"等字段，把一个image切成了很多个cell，根据这里配置，每个cell是32*32 大小，切成300个cell。然后扫描每个cell，这里面如果有需要detect的object，就标记一下。目前的配置，每个cell里只能有一个object。然后把有object的cell，当成一条数据去训练。这是一种等大小切分窗格的方式。
3. 预测的代码没有看，估计也需要这样切cell，依次过模型。



