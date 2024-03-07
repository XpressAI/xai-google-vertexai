from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import os
import requests
import shutil

from google.oauth2 import service_account

import vertexai
from vertexai.language_models import ChatModel, CodeChatModel, TextGenerationModel, CodeGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import aiplatform


@xai_component
class VertexAIAuthorize(Component):
    """Sets the Project and API key for the VertexAI client.

    ##### inPorts:
    - project: Project name id for the VertexAI API.
    - api_key: Your API key.
    - from_env: Boolean value indicating whether the API key is to be fetched from environment variables. 
    """
    project: InArg[str]
    api_key_location: InArg[str]
    location: InArg[str]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        project_id = self.project.value
        
        if self.from_env.value:
            vertexai.init(
                project=project_id,
                location=self.location.value if self.location.value else "us-central1"
            )            
        else:
            credentials = service_account.Credentials.from_service_account_file(self.api_key_location.value)
            ctx['vertexai_credential'] = credentials

            vertexai.init(
                project=project_id,
                location=self.location.value if self.location.value else "us-central1",
                credentials=credentials
            )



@xai_component
class VertexAITextGenerationModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        self.model.value = TextGenerationModel.from_pretrained(self.model_name.value)
        ctx['vertexai_model'] = self.model.value


@xai_component
class VertexAICodeGenerationModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        self.model.value = CodeGenerationModel.from_pretrained(self.model_name.value)
        ctx['vertexai_model'] = self.model.value


@xai_component
class VertexAIChatModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        self.model.value = ChatModel.from_pretrained(self.model_name.value)
        ctx['vertexai_model'] = self.model.value


@xai_component
class VertexAICodeChatModel(Component):
    model_name: InCompArg[str]
    model: OutArg[any]

    def execute(self, ctx) -> None:
        self.model.value = CodeChatModel.from_pretrained(self.model_name.value)
        ctx['vertexai_model'] = self.model.value


@xai_component
class VertexAIGenerateText(Component):
    """Generates text using a specified model or the last used model.

    ##### inPorts:
    - model: Specific model to be used for text generation.
    - prompt: The initial text to generate from.
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.

    ##### outPorts:
    - completion: The generated text.
    """

    model: InArg[any]
    prompt: InCompArg[str]
    max_tokens: InArg[int]
    temperature: InArg[float]
    top_p: InArg[float]
    top_k: InArg[int]
    completion: OutArg[str]

    def execute(self, ctx) -> None:
        parameters = {
            "max_output_tokens": self.max_tokens.value if self.max_tokens.value is not None else 1024,
            "temperature": self.temperature.value if self.temperature.value is not None else 0.2,
            "top_p": self.top_p.value if self.top_p.value is not None else 0.8,
            "top_k": self.top_k.value if self.top_k.value is not None else 40
        }
        model = self.model.value if self.model.value is not None else ctx['vertexai_model']
        response = model.predict(self.prompt.value, **parameters)
        self.completion.value = response.text
        

@xai_component
class VertexAIGenerateCode(Component):
    """Generates text using a specified model or the last used model.

    ##### inPorts:
    - model: Specific model to be used for text generation.
    - prompt: The initial text to generate from.
    - max_tokens: The maximum length of the generated text.
    - temperature: Controls randomness of the output text.

    ##### outPorts:
    - completion: The generated text.
    """

    model: InArg[any]
    prompt: InCompArg[str]
    max_tokens: InArg[int]
    temperature: InArg[float]
    completion: OutArg[str]

    def execute(self, ctx) -> None:
        parameters = {
            "max_output_tokens": self.max_tokens.value if self.max_tokens.value is not None else 1024,
            "temperature": self.temperature.value if self.temperature.value is not None else 0.2,
        }
        model = self.model.value if self.model.value is not None else ctx['vertexai_model']
        response = model.predict(self.prompt.value, **parameters)
        self.completion.value = response.text


@xai_component
class VertexAIChat(Component):
    model: InArg[any]
    conversation: InArg[object]
    context: InArg[str]
    user_prompt: InArg[str]
    response: OutArg[str]
    out_conversation: OutArg[object]
    
    def execute(self, ctx) -> None:
        if self.conversation.value is None:
            if self.context.value is not None:
                chat = self.model.value.start_chat(context=self.context.value)
            else:
                chat = self.model.value.start_chat()
        else:
            chat = self.conversation.value
            
        
        self.response.value = chat.send_message(self.user_prompt.value).text
        self.out_conversation.value = chat


@xai_component
class VertexMultimodalMakePrompt(Component):
    parts: InArg[list]
    prompt: InArg[str]
    image_path: InArg[str]
    video_path: InArg[str]
    follow_up: InArg[str]
    
    out_parts: OutArg[list]
    
    def execute(self, ctx) -> None:
        ret = [] if self.parts.value is None else self.parts.value
        
        if self.prompt.value is not None:
            ret.append(self.prompt.value)
        
        if self.image_path.value is not None:
            if self.image_path.value.endswith('.png'):
                with open(self.image_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="image/png"))
            elif self.image_path.value.endswith('.jpg') or self.image_path.value.endswith('.jpeg'):
                with open(self.image_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="image/jpeg"))
            else:
                raise Exception("Unknown image file type")
                
        if self.video_path.value is not None:
            if self.video_path.value.endswith('.mpg'):
                with open(self.video_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="video/mpeg"))
            elif self.video_path.value.endswith('.mov'):
                with open(self.video_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="video/quicktime"))
            elif self.video_path.value.endswith('.mp4'):
                with open(self.video_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="video/mp4"))
            elif self.video_path.value.endswith('.webm'):
                with open(self.video_path.value, 'rb') as f:
                    ret.append(Part.from_data(data=f.read(), mime_type="video/webm"))
            else:
                raise Exception("Unknown video file type")
        
        if self.follow_up.value is not None:
            ret.append(self.follow_up.value)
            
        self.out_parts.value = ret


@xai_component
class VertexMultimodalGenerate(Component):
    parts: InCompArg[list]
    
    max_output_tokens: InArg[int]
    temperature: InArg[float]
    top_p: InArg[int]
    top_k: InArg[int]
    
    response: OutArg[object]
    response_text: OutArg[str]
    
    def execute(self, ctx) -> None:
        model = GenerativeModel("gemini-pro-vision")
        responses = model.generate_content(
            self.parts.value,
            generation_config={
                "max_output_tokens": 2048 if self.max_output_tokens.value is None else self.max_output_tokens.value,
                "temperature": 0.4 if self.temperature.value is None else self.temperature.value,
                "top_p": 1 if self.top_p.value is None else self.top_p.value,
                "top_k": 32 if self.top_k.value is None else self.top_k.value
            }
        )

        self.response.value = responses
        self.response_text.value = responses.candidates[0].content.text

