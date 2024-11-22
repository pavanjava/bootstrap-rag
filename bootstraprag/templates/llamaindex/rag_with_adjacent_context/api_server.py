from abc import ABC

import litserve as lit


class AdjacentContextRAG(lit.LitAPI, ABC):
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


if __name__ == '__main__':
    api = AdjacentContextRAG()
    lit.LitServer(api, spec=lit.OpenAISpec())
