from semantic_router import Route

from semantic_routing_core import SemanticRouter
# Example usage
if __name__ == "__main__":
    # Initialize the SemanticRouter
    semantic_router = SemanticRouter()

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

    routes = [politics, chitchat]

    # Set up routes
    semantic_router.setup_routes(routes)

    # Test the routing
    queries = [
        "don't you love politics?",
        "how's the weather today?",
    ]

    for query in queries:
        routed_route = semantic_router.route_query(query)
        print(f"Query: '{query}' routed to: '{routed_route}'")