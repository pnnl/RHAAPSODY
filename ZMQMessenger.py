import zmq
import zmq.ssh
import json
import base64
import uuid
import time

class ZMQMessenger:  

    def __init__(self, ip, port, connection_type, identifier="", ssh_server=""):
        self.id = identifier
        self.ip = ip
        self.context = None
        self.socket = None
        self.port = port
        self.ssh_server = ssh_server

        # used for messages with a fixed reply like "ping"
        self.ReplySocket = None

        if connection_type == "PUB":
            self.ConnectAsPublisher()

        if connection_type == "SUB":
            self.ConnectAsSubscriber()

        if connection_type == "REQ":
            self.ConnectAsRequest()

        if connection_type == "REP":
            self.ConnectAsReply()

        if connection_type == "tunnelREQ":
            self.ConnectAsTunnelRequest()

        if connection_type == "tunnelREP":
            self.ConnectAsTunnelReply()

        # socket seems to take a finite time to establish
        time.sleep(1)

    def ConnectAsPublisher(self):
        self.context = zmq.Context()

        self.socket = self.context.socket(zmq.PUB)

        connect = "tcp://*:" + str(self.port)
        print(self.id + " publisher binding to:  " + connect)

        self.socket.bind(connect)

    def ConnectAsSubscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        connect = "tcp://" + self.ip + ":" + str(self.port)
        print(self.id + " subscriber connecting to:  " + connect)
        self.socket.connect(connect)
        self.socket.subscribe("")  # Subscribe to all messages, no filter

    def ConnectAsRequest(self):
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)

            connect = "tcp://*:" + str(self.port)
            print(self.id + " request binding to:  " + connect)

            self.socket.bind(connect)

    def ConnectAsTunnelRequest(self):

            connect = "tcp://" + self.ip + ":" + str(self.port)

            print(self.id + " request binding to:  " + connect)

            print("server:  ", self.ssh_server)

            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)

            print("before tunnel")
            zmq.ssh.tunnel.tunnel_connection(self.socket, "tcp://10.70.23.228:5558", "zmq@we36964ubuntuq.labnet.pnl.gov", keyfile="zmq_private")
            print("after tunnel")

    # revisit this, these are hacks I made to make it work on the other side of the tunnel
    def ConnectAsReply(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        connect = "tcp://" + self.ip + ":" + str(self.port)

        print(self.id + " reply connecting to:  " + connect)

        self.socket.connect(connect)

    def ConnectAsTunnelReply(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        connect = "tcp://*:" + str(self.port)

        print(self.id + " reply connecting to:  " + connect)


        self.socket.bind(connect)

    def CreateSocketPair(self, pairedSocket):
        pairedSocket.ReplySocket = self
        self.ReplySocket = pairedSocket

    def SendImageBinary (self, filename, flags):
        try:
            f = open(filename, 'rb')
            img_bytes = bytearray(f.read())
            f.close()

            byte_str = base64.b64encode(img_bytes)
            self.socket.send(byte_str, flags)

            print("sent as binary image:  ", filename)
            return filename
        except Exception as e:
            print(e)
            return None

    def NewHeader(self, message_type, params, u_id):
        """ header is a dictionary of meta data including the parameters for the message """

        try:
            message = {"msgType": message_type}

            # if this is a reply, send it back with the same UUID the request came with
            if u_id is not None:
                message["uuid"] = u_id
            else:
                message["uuid"] = str(uuid.uuid4())

            header = {"message": message, "parameters": params}

            return header

        except Exception as e:
            print(e)
            return

    def SendHeader(self, header, flags):
        """ header is a dictionary of meta data including the parameters for the message """

        try:
            # send the header
            self.socket.send_json(header, flags)

            print("sent header: ")
            self.PrettyPrint(header)

        except Exception as e:
            print(e)
            return

    def PrettyPrint(self, json_str):
        json_formatted_str = json.dumps(json_str, indent=2)
        print(json_formatted_str)

    def SendImage(self, filename, header):
        """ multi-part message, header and image """
        """ header is a dictionary of meta data """
        """ image is a byte array """

        try:

            print("Send Header")
            # send the header
            self.SendHeader(header, zmq.SNDMORE)

            print("Send binary image")
            # then send the image, suitably encoded
            self.SendImageBinary(filename, 0)

        except Exception as e:
            print(e)
            return


    def GetSingleImage(self, filename):
        """
        gets a single image (as a binary block) from the message queue
        stores the image locally to fName
        error returns None, success returns fName
        """

        try:
            # no message throws error
            img = self.socket.recv(zmq.NOBLOCK)

            # null message
            if img is None:
                print("no image in queue")
                return None

            image = bytearray(base64.b64decode(img))

            try:
                f = open(filename, 'wb')
                f.write(image)
                f.close()
            except:
                print("error extracting image")
                return None

            return filename

        # catches the NO_BLOCK exception
        except Exception as e:
            print(e)
            return None

    def ClearQueue(self):
        """ clears the subscribe queue"""
        data = self.socket.recv(zmq.NOBLOCK)
        while data is not None:
            data = self.socket.recv(zmq.NOBLOCK)


    def GetHeader(self, message_type):
        """ returns a header of a certain message type """

        try:
            # print("read header")
            header = self.socket.recv_json(zmq.NOBLOCK)
            if header is None:
                return None

            print("received header")
            self.PrettyPrint(header)

            if "message" not in header.keys():
                print("missing message")
                return None

            message = header["message"]

            # special case handled internally
            if message["msgType"] == "ping":
                print("was pinged")
                message["msgType"] = "pingReply"
                self.ReplySocket.SendHeader(header, 0)
                print("sent ping reply")
                return None

            if message["msgType"] != message_type:
                print("message is not: " + message_type)
                return None

            return header

        # catches the NO_BLOCK exception
        except Exception as e:
            return None


    def GetImage(self, destination_filename):
        args = dict()

        header = self.GetHeader("image")
        if header is None:
            return None

        args["header"] = header

        filename = self.GetSingleImage(destination_filename)
        if filename is None:
            print("missing incoming image")
            return None

        args["image"] = filename

        return args

    def PollFunction(self):
        """
        polls for a function call
        """
        header = self.GetHeader("function")

        return header

    def ReturnFunction(self, header):

        self.SendHeader(header, 0)


    def CallFunction(self, function, params, u_id):

        header = self.NewHeader("function", params, u_id)
        header["message"]["function"] = function
        self.SendHeader(header, 0)

        uuid = header["message"]["uuid"]
        return uuid

    def Send(self, message):
        self.socket.send(message)

    def Receive(self, flags):
        message = self.socket.recv(flags)
        return message

    def Ping(self, sender, receiver, timeout):

        pingHeader = sender.NewHeader("ping", [], None)
        print(pingHeader)
        sender.SendHeader(pingHeader, 0)

        increment = 1
        elapsed = 0

        print("awaiting ping reply from " + receiver.id)
        head = None
        while elapsed < timeout and head is None:
            elapsed += increment
            time.sleep(increment)
            head = receiver.GetHeader("pingReply")
            print(".", end="", flush=True)

        reply = (head is not None)

        print(sender.id + " ping = " + str(reply))
        return reply
