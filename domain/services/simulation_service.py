from typing import Dict, List
import json
import aiohttp
from datetime import datetime
from bson import ObjectId
from config import OPENAI_API_KEY, RETELL_API_KEY
from infrastructure.database import Database
from api.schemas.requests import CreateSimulationRequest, UpdateSimulationRequest
from fastapi import HTTPException


class SimulationService:

    def __init__(self):
        self.db = Database()

    async def create_simulation(self,
                                request: CreateSimulationRequest) -> Dict:
        """Create a new simulation"""
        try:
            # Generate prompt using OpenAI
            prompt = await self._generate_simulation_prompt(request.script)

            # Create simulation document
            simulation_doc = {
                "name": request.name,
                "divisionId": request.division_id,
                "departmentId": request.department_id,
                "type": request.type,
                "script": [s.dict() for s in request.script],
                "lastModifiedBy": request.user_id,
                "lastModified": datetime.utcnow(),
                "createdBy": request.user_id,
                "createdOn": datetime.utcnow(),
                "status": "draft",
                "version": 1,
                "prompt": prompt,
                "tags": request.tags
            }

            # Insert into database
            result = await self.db.simulations.insert_one(simulation_doc)
            return {
                "id": str(result.inserted_id),
                "status": "success",
                "prompt": prompt
            }

        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error creating simulation: {str(e)}")

    async def update_simulation(self, sim_id: str,
                                request: UpdateSimulationRequest) -> Dict:
        """Update an existing simulation"""
        try:
            # Convert string ID to ObjectId
            sim_id_object = ObjectId(sim_id)

            # Get existing simulation
            existing_sim = await self.db.simulations.find_one(
                {"_id": sim_id_object})
            if not existing_sim:
                raise HTTPException(
                    status_code=404,
                    detail=f"Simulation with id {sim_id} not found")

            # Build update document
            update_doc = {}

            # Helper function to add field if it exists in request
            def add_if_exists(field_name: str,
                              camel_case_name: str | None = None):
                value = getattr(request, field_name)
                if value is not None:
                    update_doc[camel_case_name or field_name] = value

            # Map request fields to document fields
            field_mappings = {
                "name": "name",
                "division_id": "divisionId",
                "department_id": "departmentId",
                "type": "type",
                "tags": "tags",
                "status": "status",
                "estimated_time_to_attempt_in_mins":
                "estimatedTimeToAttemptInMins",
                "key_objectives": "keyObjectives",
                "overview_video": "overviewVideo",
                "quick_tips": "quickTips",
                "voice_id": "voiceId",
                "language": "language",
                "mood": "mood",
                "voice_speed": "voice_speed",
                "prompt": "prompt",
                "simulation_completion_repetition":
                "simulationCompletionRepetition",
                "simulation_max_repetition": "simulationMaxRepetition",
                "final_simulation_score_criteria":
                "finalSimulationScoreCriteria",
                "is_locked": "isLocked",
                "version": "version",
                "assistant_id": "assistantId",
                "slides": "slides"
            }

            # Add fields from mappings
            for field, doc_field in field_mappings.items():
                add_if_exists(field, doc_field)

            # Handle special objects
            if request.script is not None:
                update_doc["script"] = [s.dict() for s in request.script]

            if request.lvl1 is not None:
                update_doc["lvl1"] = {
                    "isEnabled":
                    request.lvl1.is_enabled,
                    "enablePractice":
                    request.lvl1.enable_practice,
                    "hideAgentScript":
                    request.lvl1.hide_agent_script,
                    "hideCustomerScript":
                    request.lvl1.hide_customer_script,
                    "hideKeywordScores":
                    request.lvl1.hide_keyword_scores,
                    "hideSentimentScores":
                    request.lvl1.hide_sentiment_scores,
                    "hideHighlights":
                    request.lvl1.hide_highlights,
                    "hideCoachingTips":
                    request.lvl1.hide_coaching_tips,
                    "enablePostSimulationSurvey":
                    request.lvl1.enable_post_simulation_survey,
                    "aiPoweredPausesAndFeedback":
                    request.lvl1.ai_powered_pauses_and_feedback
                }

            if request.lvl2 is not None:
                update_doc["lvl2"] = {"isEnabled": request.lvl2.is_enabled}

            if request.lvl3 is not None:
                update_doc["lvl3"] = {"isEnabled": request.lvl3.is_enabled}

            if request.simulation_scoring_metrics is not None:
                update_doc["simulationScoringMetrics"] = {
                    "isEnabled": request.simulation_scoring_metrics.is_enabled,
                    "keywordScore":
                    request.simulation_scoring_metrics.keyword_score,
                    "clickScore":
                    request.simulation_scoring_metrics.click_score
                }

            if request.sim_practice is not None:
                update_doc["simPractice"] = {
                    "isUnlimited": request.sim_practice.is_unlimited,
                    "preRequisiteLimit":
                    request.sim_practice.pre_requisite_limit
                }

            # Create LLM and Agent if prompt is provided
            if request.prompt is not None:
                # Create Retell LLM
                llm_response = await self._create_retell_llm(request.prompt)
                update_doc["llmId"] = llm_response["llm_id"]

                # Create Retell Agent
                agent_response = await self._create_retell_agent(
                    llm_response["llm_id"], request.voice_id
                    or "11labs-Adrian")
                update_doc["agentId"] = agent_response["agent_id"]

            # Add metadata
            update_doc["lastModified"] = datetime.utcnow()
            update_doc["lastModifiedBy"] = request.user_id

            # Update database
            result = await self.db.simulations.update_one(
                {"_id": sim_id_object}, {"$set": update_doc})

            if result.modified_count == 0:
                raise HTTPException(status_code=500,
                                    detail="Failed to update simulation")

            return {"id": sim_id, "status": "success"}

        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error updating simulation: {str(e)}")

    async def _create_retell_llm(self, prompt: str) -> Dict:
        """Create a new Retell LLM"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {RETELL_API_KEY}',
                    'Content-Type': 'application/json'
                }

                data = {"general_prompt": prompt}

                async with session.post(
                        'https://api.retellai.com/create-retell-llm',
                        headers=headers,
                        json=data) as response:
                    if response.status != 201:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to create Retell LLM")

                    return await response.json()

        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error creating Retell LLM: {str(e)}")

    async def _create_retell_agent(self, llm_id: str, voice_id: str) -> Dict:
        """Create a new Retell Agent"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {RETELL_API_KEY}',
                    'Content-Type': 'application/json'
                }

                data = {
                    "response_engine": {
                        "llm_id": llm_id,
                        "type": "retell-llm"
                    },
                    "voice_id": voice_id
                }

                async with session.post(
                        'https://api.retellai.com/create-agent',
                        headers=headers,
                        json=data) as response:
                    if response.status != 201:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to create Retell Agent")

                    return await response.json()

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error creating Retell Agent: {str(e)}")

    async def _generate_simulation_prompt(self, script: List[Dict]) -> str:
        """Generate simulation prompt using OpenAI"""
        try:
            # Convert script to conversation format for prompt
            conversation = "\n".join(
                [f"{s.role}: {s.script_sentence}" for s in script])

            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                }

                data = {
                    "model":
                    "gpt-4o",
                    "messages": [{
                        "role":
                        "system",
                        "content":
                        "Create a detailed prompt for a customer service simulation based on the following conversation. The prompt should help generate realistic customer responses that match the conversation flow and context. Consider the sequence of interactions and maintain consistency with the original conversation."
                    }, {
                        "role": "user",
                        "content": conversation
                    }]
                }

                async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status,
                                            detail=response)

                    result = await response.json()
                    return result['choices'][0]['message']['content']

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating simulation prompt: {str(e)}")

    async def start_audio_simulation_preview(self, sim_id: str,
                                             user_id: str) -> Dict:
        """Start an audio simulation preview"""
        try:
            # Convert string ID to ObjectId
            sim_id_object = ObjectId(sim_id)

            # Get simulation
            simulation = await self.db.simulations.find_one(
                {"_id": sim_id_object})
            if not simulation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Simulation with id {sim_id} not found")

            # Get agent_id
            agent_id = simulation.get("agentId")
            if not agent_id:
                raise HTTPException(
                    status_code=400,
                    detail="Simulation does not have an agent configured")

            # Create web call
            web_call = await self._create_web_call(agent_id)

            return {"access_token": web_call["access_token"]}

        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error starting audio simulation preview: {str(e)}")

    async def _create_web_call(self, agent_id: str) -> Dict:
        """Create a web call using Retell API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {RETELL_API_KEY}',
                    'Content-Type': 'application/json'
                }

                data = {"agent_id": agent_id}

                async with session.post(
                        'https://api.retellai.com/v2/create-web-call',
                        headers=headers,
                        json=data) as response:
                    print(await response.json())
                    if response.status != 201:
                        raise HTTPException(status_code=response.status,
                                            detail="Failed to create web call")

                    return await response.json()

        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Error creating web call: {str(e)}")
