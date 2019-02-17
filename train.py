import argparse

import helper as h

#
# Get command line arguments
#
parser =argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data Directory", required=True, type=str)
parser.add_argument("--arch", help="Choose VGG13 or VGG16. Default is VGG13", required=False, type=str, default="VGG13")
parser.add_argument("--learning_rate", help="Learning rate. Default is 0.001", required=False, type=float, default=0.001)
parser.add_argument("--gpu",help="Optional to run on gpu if available. Strongly recommended!", action='store_true')
parser.add_argument("--hidden_units",help="# of hidden units. Default is 512",type=int,default=512)
parser.add_argument("--output_units",help="# of output categories",type=int,default=102)
parser.add_argument("--epochs",help="# of epochs. Default is 5", type=int, default=5, required=False)
parser.add_argument("--save_dir", type=str, default=".", required=False
                    , help="Directory (needs to exist) to store saved model. Default is current directory."
                    )
namespace = parser.parse_args()
if namespace.gpu == True:
    device_type = 'cuda'
else:
    device_type = 'cpu'
h.trainAndCheckpointModel(
                          h.initializePretrainedModel(
                              namespace.arch
                              ,namespace.hidden_units
                              ,output_units=namespace.output_units
                          )
                          ,namespace.arch
                          ,namespace.data_dir
                          ,save_dir = namespace.save_dir
                          ,epochs=namespace.epochs
                          ,lr=namespace.learning_rate
                          ,device_type=device_type
                          )
