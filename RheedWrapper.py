import os
import sys
import math
import json
import time
import gc
from ZMQMessenger import ZMQMessenger

# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(".")
from embeddings import EmbeddingModel
from auto_rheeder import AutoRHEEDer
from change_detection import ChangepointDetection
from graph import GraphClustering
from segment import Segment

root_directory = "c:\\"
Rheed_subscriber = None

# pnl_server='WE37491.pnl.gov'
pnl_server='130.20.173.71'
labnet_server='WE27790.labnet.pnl.gov'

def ConnectRheed(publisher_port, subscriber_port, subscriber_ip):
    global Rheed_subscriber
    global Rheed_publisher

    Rheed_publisher = ZMQMessenger("localhost", publisher_port, "PUB", "rheed")
    Rheed_subscriber = ZMQMessenger(subscriber_ip, subscriber_port, "SUB", "rheed")
    Rheed_publisher.CreateSocketPair(Rheed_subscriber)

def MessageHandler(header):
    print("PyJEM handle message")

    try:
        print("MESSAGE:")
        print(header)

        if "message" not in header.keys():
            print("missing message")
            return

        # clear queue if called
        if header['message']['msgType']=='Reset':
            Rheed_publisher.ClearQueue()
            print('QUEUE CLEARED')
            return

        # call to analysis 
        reply_header = rheed_analysis.receive_message(header)

        # create the return message
        print("REPLY:")
        print(reply_header)

        # send the return message
        Rheed_publisher.SendHeader(reply_header, 0)


    except Exception as e:
        print(e)


def Version():
    return "1/30/24"

def main():

    print("RheedWrapper version: " + Version())

    # start the messenger
    # note these are the reverse of the control connections
    # publisher_port, subscriber_port, subscriber_ip
    ConnectRheed(5556, 5555, pnl_server)

    count = 0
    while True:

        header = Rheed_subscriber.GetHeader("RawImage")

        while header is not None:
            MessageHandler(header)
            header = Rheed_subscriber.GetHeader("RawImage")
            count+=1
            if count%10==0:
                gc.collect()

        time.sleep(.5)


if __name__ == "__main__":
    gc.disable()
    
    # initialize analysis
    rheed_analysis = AutoRHEEDer(
            root="/mnt/atscale/TestExp",
            data_processor=EmbeddingModel(),
            change_detector=ChangepointDetection(cost_threshold=0.05, window_size=300, min_time_between_changepoints=10),
            classifier=GraphClustering(resolution=1, seed=123),
            segmenter=Segment(),
            starting_period=30,
            max_embeddings=2101,
            max_steps=2100,
            )
    main()
