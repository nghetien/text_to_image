from openai import OpenAI

from models.text_to_image import TextToImage


class OpenAIGPT(TextToImage):
    client: OpenAI

    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def text_to_image(self, prompt: str, number_images: int) -> list[str]:
        responses = []
        for i in range(number_images):
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            responses.append(image_url)
        return responses
