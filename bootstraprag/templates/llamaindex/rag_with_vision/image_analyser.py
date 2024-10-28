from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from llama_index.multi_modal_llms.huggingface import HuggingFaceMultiModal
from llama_index.core.schema import ImageDocument
import os
from typing import Optional, List, Union


class ImageAnalyzer:
    """
    A class to analyze images using the Qwen2-VL-2B-Instruct model from HuggingFace.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", max_new_tokens: int = 512):
        """
        Initialize the ImageAnalyzer.

        Args:
            model_name (str): The name of the HuggingFace model to use
            max_new_tokens (int): Maximum number of tokens to generate
        """
        # Load environment variables
        load_dotenv(find_dotenv())

        # Login to HuggingFace
        self._login()

        # Initialize the model
        self.model = HuggingFaceMultiModal.from_model_name(
            model_name,
            max_new_tokens=max_new_tokens
        )

    def _login(self) -> None:
        """
        Login to HuggingFace using the token from environment variables.

        Raises:
            ValueError: If HF_TOKEN is not found in environment variables
        """
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN not found in environment variables")
        login(token=token)

    def analyze_image(self,
                      image_path: str,
                      prompt: str = "Understand the Image and give the detailed summary.",
                      additional_images: Optional[List[str]] = None) -> str:
        """
        Analyze an image or multiple images with a given prompt.

        Args:
            image_path (str): Path to the main image file
            prompt (str): The prompt to use for analysis
            additional_images (List[str], optional): List of paths to additional images

        Returns:
            str: The analysis response from the model

        Raises:
            FileNotFoundError: If the image file(s) cannot be found
        """
        # Validate main image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Create list of image documents
        image_documents = [ImageDocument(image_path=image_path)]

        # Add additional images if provided
        if additional_images:
            for add_image_path in additional_images:
                if not os.path.exists(add_image_path):
                    raise FileNotFoundError(f"Additional image not found at path: {add_image_path}")
                image_documents.append(ImageDocument(image_path=add_image_path))

        # Generate response
        print('Started analyzing...')
        response = self.model.complete(prompt, image_documents=image_documents)

        return response.text

    def batch_analyze(self,
                      image_paths: List[str],
                      prompts: Union[str, List[str]]) -> List[str]:
        """
        Analyze multiple images with either a single prompt or multiple prompts.

        Args:
            image_paths (List[str]): List of paths to image files
            prompts (Union[str, List[str]]): Single prompt or list of prompts matching images

        Returns:
            List[str]: List of analysis responses

        Raises:
            ValueError: If number of prompts doesn't match number of images when using multiple prompts
        """
        if isinstance(prompts, list) and len(prompts) != len(image_paths):
            raise ValueError("Number of prompts must match number of images when using multiple prompts")

        results = []
        for idx, image_path in enumerate(image_paths):
            current_prompt = prompts[idx] if isinstance(prompts, list) else prompts
            try:
                result = self.analyze_image(image_path, current_prompt)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing image {image_path}: {str(e)}")
                results.append(None)

        return results
