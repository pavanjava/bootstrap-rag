import litserve as lit


class IntrospectionAgentAPI(lit.LitAPI):
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


if __name__ == "__main__":
    api = IntrospectionAgentAPI()
    lit.LitServer(lit_api=api, spec=lit.OpenAISpec())
