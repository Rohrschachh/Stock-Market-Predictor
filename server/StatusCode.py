from enum import Enum

class ServerStatusCodes(Enum):
    SUCCESS = 200
    TRAININGMODEL = 503
    HEAVYTRAFFIC = 502
    UNKNOWN = 500
    SERVERERROR = 504
    NORESOURCE = 403
    BADREQUEST = 400
    NOTFOUND = 404
    NOTIMPLEMENTED = 501
    