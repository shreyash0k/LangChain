from base64 import b64encode
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    model = "claude-haiku-4-5"
)

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe this image'},
        # use base64 encoding to send the image from resources folder
        {'type': 'image', 
        'base64': b64encode(open('resources/iphone17pro.jpg', 'rb').read()).decode('utf-8'),
        'mime_type': 'image/jpeg',

        }

        
    ]
}

response = model.invoke([message])
print(response.content)