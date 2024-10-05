from multimodal_search import MultiModalSearch


documents = [{"caption": "An Architecture describing MediaQ platform",
              "image": "images/MediaQ.png"},
             {"caption": "An Architecture describing the Advanced RAG",
              "image": "images/adv-RAG.png"},
             {"caption": "An Architecture describing Vision based RAG",
              "image": "images/VisionRAG.png"}
             ]

mm_search = MultiModalSearch(documents=documents)
# mm_search.search_image_by_text(user_query="propose an advanced RAG architecture")
comment = mm_search.search_text_by_image(image_path='images/VisionRAG.png')
print(comment)






