from abc import ABC

from semantic_router import Route

from semantic_routing_core import SemanticRouter
import litserve as ls
import os


class SemanticRoutingAPI(ls.LitAPI, ABC):
    def __init__(self):
        self.semantic_routing_core = None
        # Define routes
        politics = Route(
            name="politics",
            utterances=[
                "isn't politics the best thing ever",
                "why don't you tell me about your political opinions",
                "don't you just love the president",
                "they're going to destroy this country!",
                "they will save the country!",
            ],
        )

        chitchat = Route(
            name="chitchat",
            utterances=[
                "how's the weather today?",
                "how are things going?",
                "lovely weather today",
                "the weather is horrendous",
                "let's go to the chippy",
            ],
        )

        self.routes = [politics, chitchat]

    def setup(self, device):
        self.semantic_routing_core = SemanticRouter()
        # Set up routes
        self.semantic_routing_core.setup_routes(self.routes)

    def decode_request(self, request, **kwargs):
        return request['question']

    def predict(self, query, **kwargs):
        return self.semantic_routing_core.route_query(query=query)

    def encode_response(self, output, **kwargs):
        return {'response': output}


if __name__ == '__main__':
    api = SemanticRoutingAPI()
    server = ls.LitServer(lit_api=api, api_path='/api/v1/chat-completion',
                          workers_per_device=int(os.environ.get('LIT_SERVER_WORKERS_PER_DEVICE')))
    server.run(port=os.environ.get('LIT_SERVER_PORT'))
