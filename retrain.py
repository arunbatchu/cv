import argparse

import helper as h

#
# Get command line arguments
#
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Data Directory", required=True, type=str)
parser.add_argument("--learning_rate", help="Learning rate. Default is 0.001", required=False, type=float,
                    default=0.001)
parser.add_argument("--gpu", help="Optional to run on gpu if available. Strongly recommended!", action='store_true')
parser.add_argument("--epochs", help="# of epochs. Default is 5", type=int, default=5, required=False)

namespace = parser.parse_args()
previously_trained_model = h.retrieveModelFromCheckpoint("checkpoint.pth")

device_type = h.get_device_type(namespace.gpu)
# TODO : remove hardcoded assumption of checkpoint directory
model =  h.trainAndCheckpointModel(previously_trained_model
                          , previously_trained_model.arch
                          , namespace.data_dir, "."
                          , epochs_completed=previously_trained_model.epochs_completed
                          , epochs=namespace.epochs
                          , lr=namespace.learning_rate
                          , device_type=device_type)
h.test_trained_network(model,h.createTestingDataloader(namespace.data_dir))