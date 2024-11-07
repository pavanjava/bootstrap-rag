from abc import ABC
import litserve as ls


class SemanticRoutingAPI(ls.LitAPI, ABC):
    def __init__(self):
        pass

    def setup(self, device):
        pass

    def decode_request(self, request, **kwargs):
        pass

    def predict(self, x, **kwargs):
        pass

    def encode_response(self, output, **kwargs):
        pass
