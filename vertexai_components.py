from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import vertexai
from vertexai.language_models import ChatModel, CodeChatModel, TextGenerationModel, CodeGenerationModel
import os


@xai_component
class VertexAIAuthorize(Component):
    project: InArg[str]
    api_key: InArg[secret]
    location: InArg[str]
    from_env: InArg[bool]

    def execute(self, ctx) -> None:
        project_id = self.project.value
        
        # TODO: Figure out how auth should work.
        if self.from_env.value:
            api_key = os.getenv("VERTEXAI_API_KEY")
        else:
            api_key = self.api_key.value
        
        vertexai.init(
            project=project_id,
            location=self.location.value if self.location.value else "us-central1"
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
