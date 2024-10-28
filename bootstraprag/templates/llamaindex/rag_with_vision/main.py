from image_analyser import ImageAnalyzer
# Basic usage
analyzer = ImageAnalyzer()
result = analyzer.analyze_image("img.png")
print(result)

# Custom prompt
result = analyzer.analyze_image(
    "img.png",
    prompt="Describe the main objects in this image"
)

# Batch analysis
# image_paths = ["img1.png", "img2.png", "img3.png"]
# results = analyzer.batch_analyze(
#     image_paths,
#     prompts="Analyze this image"
# )