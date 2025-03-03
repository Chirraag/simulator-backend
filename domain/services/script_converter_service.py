from fastapi import UploadFile, HTTPException
from typing import List, Dict
import json
import aiohttp
import PyPDF2
import io
import docx
from config import (DEEPGRAM_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME,
                    AZURE_OPENAI_KEY, AZURE_OPENAI_BASE_URL)
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings, )
from domain.plugins.deepgram_plugin import DeepgramPlugin
from typing import List
from pydantic import BaseModel


class ScriptItem(BaseModel):
    role: str
    message: str


class MyResponseSchema(BaseModel):
    script: List[ScriptItem]


class ScriptConverterService:

    def __init__(self):
        # Initialize Semantic Kernel
        self.kernel = Kernel()

        # Add Azure OpenAI service
        self.chat_completion = AzureChatCompletion(
            service_id="azure_gpt4",
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            endpoint=AZURE_OPENAI_BASE_URL,
            api_key=AZURE_OPENAI_KEY)
        self.kernel.add_service(self.chat_completion)

        # Add fallback strategy

        # Add Deepgram plugin
        self.deepgram_plugin = DeepgramPlugin(DEEPGRAM_API_KEY)
        self.kernel.add_plugin(self.deepgram_plugin, "DeepgramPlugin")

        # Example JSON schema that expects an object with a "script" array of objects

        # Configure execution settings
        self.execution_settings = AzureChatPromptExecutionSettings(
            service_id="azure_gpt4", # Add config values
            ai_model_id=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.7, #test temperature
            top_p=1.0,
            max_tokens=2000,
            response_format=MyResponseSchema)

    async def convert_audio_to_script(
            self, user_id: str,
            audio_file: UploadFile) -> List[Dict[str, str]]:
        """
        Convert audio to script using Deepgram plugin.
        """
        try:
            # Read the audio file content
            audio_content = await audio_file.read()

            # Get the transcribe function from the plugin
            transcribe_function = self.kernel.plugins["DeepgramPlugin"][
                "transcribe_audio"]

            # Use Deepgram plugin to transcribe
            transcript = await self.deepgram_plugin.transcribe_audio(
                audio_content)

            print(str(transcript))

            # Convert transcript to conversation format
            return await self._convert_transcript_to_conversation_format(
                str(transcript))

        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error processing audio: {str(e)}")

    async def convert_text_to_script(self, user_id: str,
                                     prompt: str) -> List[Dict[str, str]]:
        """
        Convert text to conversation script using Azure OpenAI.
        """
        try:
            return await self._convert_prompt_to_conversation_format(prompt)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting text to script: {str(e)}")

    async def convert_file_to_script(self, user_id: str,
                                     file: UploadFile) -> List[Dict[str, str]]:
        """
        Convert file content to conversation script using Azure OpenAI.
        """
        try:
            content = await self._extract_text_from_file(file)
            return await self._convert_prompt_to_conversation_format(content)
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error processing file: {str(e)}")

    async def _extract_text_from_file(self, file: UploadFile) -> str:
        """
        Extract text content from different file types
        """
        content = await file.read()

        if file.content_type == 'text/plain':
            return content.decode('utf-8')
        elif file.content_type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return ' '.join(page.extract_text() for page in pdf_reader.pages)
        elif file.content_type in [
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]:
            doc = docx.Document(io.BytesIO(content))
            return ' '.join(paragraph.text for paragraph in doc.paragraphs)
        else:
            raise HTTPException(status_code=400,
                                detail="Unsupported file type")

    async def _convert_prompt_to_conversation_format(
            self, content: str) -> List[Dict[str, str]]:
        """
        Convert content to conversation format using Azure OpenAI
        """
        try:
            history = ChatHistory()

            # Add system message
            history.add_system_message(
                "Convert the following text into a natural conversation between a user and an assistant. "
                "Return the result as a JSON object with a 'script' array containing objects with 'role' and "
                "'message' fields. The conversation should flow naturally and make sense. There are only two "
                "roles 'Customer' and 'Trainee' in the conversation. The user is always the Customer and the "
                "Trainee is always the assistant.")

            # Add user content
            history.add_user_message(content)

            # Get response from Azure OpenAI
            result = await self.chat_completion.get_chat_message_content(
                history, settings=self.execution_settings)

            # Parse the response
            try:
                conversation = json.loads(str(result))
                return conversation.get('script', [])
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse Azure OpenAI response")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting to conversation format: {str(e)}")

    async def _convert_transcript_to_conversation_format(
            self, content: str) -> List[Dict[str, str]]:
        """
        Convert transcript to conversation format using Azure OpenAI
        """
        try:
            history = ChatHistory()

            # Add system message
            history.add_system_message(
                "Convert the following phoneCall transcript of two callers marked as Speaker 0 and Speaker 1 "
                "into a conversation between a Customer and Trainee. Return the result as a JSON object with "
                "a 'script' array containing objects with 'role' and 'message' fields. Also, do not add "
                "conversation from your end. As per transcript you can start conversation with either Customer "
                "or Trainee. Do not miss any important line from transcript.")

            # Add user content
            history.add_user_message(content)

            # Get response from Azure OpenAI
            result = await self.chat_completion.get_chat_message_content(
                history, settings=self.execution_settings)

            print("Azure response:", result)

            # Parse the response
            try:
                conversation = json.loads(str(result))
                return conversation.get('script', [])
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse Azure OpenAI response")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting to conversation format: {str(e)}")
