import tensorflow as tf
import numpy as np
import cv2
import sys

def Predict(x_arr):
    x_arr = cv2.resize(x_arr, (64, 64))
    x_arr = np.resize(x_arr, (1, 64, 64, 3))   
    labels = {1:"0",2:"1",3:"2",4:"3",5:"4",6:"5",7:"6",8:"7",9:"8",10:"9",
              11:"A",12:"B",13:"C",14:"D",15:"E",16:"F",17:"G",18:"H",19:"I",20:"J",
              21:"K",22:"L",23:"M",24:"N",25:"O",26:"P",27:"Q",28:"R",29:"S",30:"T",
              31:"U",32:"V",33:"W",34:"X",35:"Y",36:"Z",37:"a",38:"b",39:"c",40:"d",
              41:"e",42:"f",43:"g",44:"h",45:"i",46:"j",47:"k",48:"l",49:"m",50:"n",
              51:"o",52:"p",53:"q",54:"r",55:"s",56:"t",57:"u",58:"v",59:"w",60:"x",
              61:"y",62:"z",0:"None"
             }
    tf.reset_default_graph()
    checkpoint_path="my-test-model" #Write your path for .meta file
    with tf.Session() as sess:

    ## Load the entire model previuosly saved in a checkpoint
    #    print("Load the model from path", checkpoint_path)
        the_Saver = tf.train.import_meta_graph(checkpoint_path+".meta")
        the_Saver.restore(sess, checkpoint_path)

        ## Identify the predictor of the Tensorflow graph
        predict_op = tf.get_collection('predict_op')[0]

        ## Identify the restored Tensorflow graph
        dataFlowGraph = tf.get_default_graph()

        ## Identify the input placeholder to feed the images into as defined in the model
        x = dataFlowGraph.get_tensor_by_name("X:0")

        ## Predict the image category
        prediction = sess.run(predict_op, feed_dict = {x: x_arr})
        val = int(np.squeeze(prediction))
        print("nThe predicted image class is:", labels[val])
        return labels[val]

