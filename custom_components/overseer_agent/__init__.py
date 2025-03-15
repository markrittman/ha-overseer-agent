import logging
import voluptuous as vol
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Deque
import asyncio
from collections import deque
import json
import time
import os
import shutil

from homeassistant.core import HomeAssistant, State, Event, Context, callback, ServiceCall
from homeassistant.helpers.typing import ConfigType
from homeassistant.helpers import config_validation as cv
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.helpers.entity import Entity
from homeassistant.components import conversation, websocket_api
from homeassistant.components.conversation import ConversationInput, ConversationResult
try:
    # For newer versions of Home Assistant
    from homeassistant.components.conversation.agent import AbstractConversationAgent
except ImportError:
    # For older versions of Home Assistant
    from homeassistant.components.conversation import AbstractConversationAgent
from homeassistant.components.http import HomeAssistantView
from homeassistant.components.websocket_api import websocket_command
from homeassistant.components.lovelace.resources import ResourceStorageCollection
from homeassistant.helpers.event import async_call_later

DOMAIN = "overseer_agent"
_LOGGER = logging.getLogger(__name__)

# Handle different versions of LangChain
try:
    # For newer versions of LangChain (>= 0.1.0)
    # Import VertexAI from the dedicated package
    try:
        from langchain_google_vertexai import VertexAI
        _LOGGER.info("Using langchain_google_vertexai for VertexAI")
    except ImportError:
        from langchain_community.llms import VertexAI
        _LOGGER.info("Using langchain_community for VertexAI")
        
    from langchain.agents import initialize_agent, AgentType
    # Use newer memory imports
    try:
        from langchain.memory import ConversationBufferMemory
        _LOGGER.info("Using legacy ConversationBufferMemory")
    except (ImportError, DeprecationWarning):
        from langchain_core.memory import ConversationBufferMemory
        _LOGGER.info("Using langchain_core.memory.ConversationBufferMemory")
        
    from langchain.tools import BaseTool
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    _LOGGER.info("Using newer LangChain API (>= 0.1.0)")
except ImportError:
    try:
        # For older versions of LangChain (< 0.1.0)
        from langchain.llms import VertexAI
        from langchain.agents import initialize_agent, AgentType
        from langchain.memory import ConversationBufferMemory
        from langchain.tools import BaseTool
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        _LOGGER.info("Using older LangChain API (< 0.1.0)")
    except ImportError:
        _LOGGER.error("Failed to import LangChain. Make sure it's installed correctly.")
        VertexAI = None
        initialize_agent = None
        AgentType = None
        ConversationBufferMemory = None
        BaseTool = None
        LLMChain = None
        PromptTemplate = None

# Service constants
SERVICE_QUERY = "query"
SERVICE_ANALYZE_ENTITY = "analyze_entity"
SERVICE_ANALYZE_DOMAIN = "analyze_domain"
SERVICE_CLEAR_INSIGHTS = "clear_insights"

# WebSocket commands
WS_TYPE_OVERSEER_INSIGHTS = f"{DOMAIN}/insights"
WS_TYPE_OVERSEER_SUBSCRIBE = f"{DOMAIN}/subscribe"

# Configuration constants
DEFAULT_UPDATE_INTERVAL = 30
MAX_HISTORY_ITEMS = 100
MAX_INSIGHT_ITEMS = 50

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Required("google_cloud_project_id"): cv.string,
        vol.Required("google_cloud_credentials"): cv.string,
        vol.Optional("include_domains", default=[]): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional("exclude_domains", default=[]): vol.All(cv.ensure_list, [cv.string]),
        vol.Optional("update_interval", default=DEFAULT_UPDATE_INTERVAL): cv.positive_int,
        vol.Optional("enable_conversation", default=True): cv.boolean,
        vol.Optional("max_insights", default=MAX_INSIGHT_ITEMS): cv.positive_int,
        vol.Optional("conversational_style", default="thoughtful"): vol.In(
            ["thoughtful", "analytical", "friendly", "concise"]
        ),
    })
}, extra=vol.ALLOW_EXTRA)

class InsightEntry:
    """Class to represent an insight entry."""
    
    def __init__(self, content: str, source: str, timestamp: Optional[datetime] = None):
        """Initialize the insight entry."""
        self.content = content
        self.source = source
        self.timestamp = timestamp or datetime.now()
        
    def as_dict(self) -> Dict[str, Any]:
        """Return the insight as a dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

class StateSummaryTool(BaseTool):
    """Tool for summarizing the current state of the smart home."""
    
    name: str = "state_summary"
    description: str = "Summarize the current state of the smart home system and its devices"

    def __init__(self, state_history: Dict[str, Any]):
        self.state_history = state_history
        super().__init__()

    def _run(self, query: str) -> str:
        """Summarize the current state."""
        summaries = []
        for entity_id, state_data in self.state_history.items():
            summaries.append(f"{entity_id}: {state_data['state']}")
        return "\n".join(summaries)

    async def _arun(self, query: str) -> str:
        """Run async version."""
        return self._run(query)

class EntityQueryTool(BaseTool):
    """Tool for querying specific entities or domains."""
    
    name: str = "entity_query"
    description: str = "Get detailed information about specific entities or domains in the smart home"

    def __init__(self, state_history: Dict[str, Any], hass: HomeAssistant):
        self.state_history = state_history
        self.hass = hass
        super().__init__()

    def _run(self, query: str) -> str:
        """Query specific entities or domains."""
        parts = query.split()
        if not parts:
            return "Please specify an entity ID or domain to query."
            
        target = parts[0].lower()
        
        # Check if querying a specific entity
        if target in self.state_history:
            entity_data = self.state_history[target]
            return f"Entity: {target}\nState: {entity_data['state']}\nLast Changed: {entity_data['last_changed']}\nAttributes: {entity_data['attributes']}"
        
        # Check if querying a domain
        domain_entities = [
            (entity_id, data) for entity_id, data in self.state_history.items()
            if entity_id.split('.')[0] == target
        ]
        
        if domain_entities:
            results = [f"Domain: {target} - {len(domain_entities)} entities found:"]
            for entity_id, data in domain_entities:
                results.append(f"- {entity_id}: {data['state']}")
            return "\n".join(results)
            
        return f"No entities found matching '{target}'."

    async def _arun(self, query: str) -> str:
        """Run async version."""
        return self._run(query)

class OverseerAgent:
    """Main class for the Overseer Agent."""

    def __init__(self, hass: HomeAssistant, config: dict):
        """Initialize the Overseer Agent."""
        self.hass = hass
        self.config = config
        self.state_history: Dict[str, Any] = {}
        self.event_queue = asyncio.Queue()
        self.processing = False
        self.llm = None
        self.agent = None
        self.memory = None
        self.conversation_agent = None
        self.insights: Deque[InsightEntry] = deque(maxlen=config.get("max_insights", MAX_INSIGHT_ITEMS))
        self.subscribers = set()
        self.conversational_style = config.get("conversational_style", "thoughtful")
        
    async def async_start(self):
        """Start the overseer agent."""
        try:
            # Initialize Vertex AI
            self.llm = VertexAI(
                project=self.config["google_cloud_project_id"],
                credentials_path=self.config["google_cloud_credentials"],
                model_name="gemini-2.0-flash",
                temperature=0.1,
                max_output_tokens=1024
            )
            
            # Initialize tools
            tools = [
                StateSummaryTool(self.state_history),
                EntityQueryTool(self.state_history, self.hass)
            ]
            
            # Initialize the memory
            try:
                # Try newer memory API
                from langchain_core.memory import ChatMessageHistory
                from langchain_core.messages import HumanMessage, AIMessage
                
                # Initialize with the newer API
                message_history = ChatMessageHistory()
                self.memory = ConversationBufferMemory(
                    chat_memory=message_history,
                    return_messages=True
                )
                _LOGGER.info("Using newer memory API")
            except ImportError:
                # Fall back to older memory API
                self.memory = ConversationBufferMemory(return_messages=True)
                _LOGGER.info("Using older memory API")
            
            # Initialize the agent
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True
            )
            
            # Start listening for state changes
            self.hass.bus.async_listen(EVENT_STATE_CHANGED, self._handle_state_change)
            
            # Start the processing loop
            self.processing = True
            self.hass.loop.create_task(self._process_events())
            
            # Register conversation agent if enabled
            if self.config.get("enable_conversation", True):
                await self._register_conversation_agent()
                
            # Register services
            self._register_services()
            
            # Register websocket commands
            self._register_websocket_commands()
            
            # Add initial insight
            self._add_insight(
                "I'm now online and monitoring your home. I'll keep track of device states and provide insights about what's happening.",
                "system"
            )
            
            _LOGGER.info("Overseer Agent started successfully")
            return True
            
        except Exception as e:
            _LOGGER.error(f"Failed to start Overseer Agent: {str(e)}")
            return False

    async def _register_conversation_agent(self):
        """Register the conversation agent."""
        try:
            # Create conversation agent
            self.conversation_agent = OverseerConversationAgent(self)
            
            # Register the agent with Home Assistant
            await self.conversation_agent.async_register()
            
            _LOGGER.info("Registered conversation agent")
        except Exception as e:
            _LOGGER.error(f"Failed to register conversation agent: {str(e)}")

    def _register_services(self):
        """Register services for the component."""
        self.hass.services.async_register(
            DOMAIN, 
            SERVICE_QUERY, 
            self._handle_query_service, 
            schema=vol.Schema({
                vol.Required("query"): cv.string,
            })
        )
        
        self.hass.services.async_register(
            DOMAIN, 
            SERVICE_ANALYZE_ENTITY, 
            self._handle_analyze_entity_service, 
            schema=vol.Schema({
                vol.Required("entity_id"): cv.entity_id,
            })
        )
        
        self.hass.services.async_register(
            DOMAIN, 
            SERVICE_ANALYZE_DOMAIN, 
            self._handle_analyze_domain_service, 
            schema=vol.Schema({
                vol.Required("domain"): cv.string,
            })
        )
        
        self.hass.services.async_register(
            DOMAIN, 
            SERVICE_CLEAR_INSIGHTS, 
            self._handle_clear_insights_service, 
            schema=vol.Schema({})
        )
        
        _LOGGER.info("Registered Overseer Agent services")
        
    def _register_websocket_commands(self):
        """Register websocket commands."""
        websocket_api.async_register_command(
            self.hass,
            WS_TYPE_OVERSEER_INSIGHTS,
            self._handle_websocket_insights,
            websocket_api.BASE_COMMAND_MESSAGE_SCHEMA.extend({
                vol.Required("type"): WS_TYPE_OVERSEER_INSIGHTS,
                vol.Optional("count"): vol.Coerce(int),
            })
        )
        
        websocket_api.async_register_command(
            self.hass,
            WS_TYPE_OVERSEER_SUBSCRIBE,
            self._handle_websocket_subscribe,
            websocket_api.BASE_COMMAND_MESSAGE_SCHEMA.extend({
                vol.Required("type"): WS_TYPE_OVERSEER_SUBSCRIBE,
            })
        )
        
        _LOGGER.info("Registered Overseer Agent websocket commands")

    def should_track_entity(self, entity_id: str) -> bool:
        """Determine if an entity should be tracked based on configuration."""
        domain = entity_id.split('.')[0]
        
        if self.config["include_domains"] and domain not in self.config["include_domains"]:
            return False
            
        if domain in self.config["exclude_domains"]:
            return False
            
        return True

    async def _handle_state_change(self, event: Event):
        """Handle state change events."""
        entity_id = event.data.get("entity_id")
        
        if not self.should_track_entity(entity_id):
            return
            
        new_state = event.data.get("new_state")
        if new_state is None:
            return

        # Update state history
        self.state_history[entity_id] = {
            "state": new_state.state,
            "attributes": dict(new_state.attributes),
            "last_changed": new_state.last_changed.isoformat(),
        }
        
        # Add to processing queue
        await self.event_queue.put({
            "type": "state_changed",
            "entity_id": entity_id,
            "state": new_state.state,
            "previous_state": event.data.get("old_state", {}).get("state"),
            "timestamp": new_state.last_changed.isoformat()
        })

    async def _process_events(self):
        """Process events from the queue."""
        while self.processing:
            try:
                # Get events in batches
                events = []
                try:
                    while len(events) < 5:  # Process up to 5 events at once
                        event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                        events.append(event)
                except asyncio.TimeoutError:
                    if not events:
                        continue

                # Analyze events with the LLM
                event_descriptions = "\n".join(
                    f"Entity {e['entity_id']} changed from {e['previous_state']} to {e['state']}"
                    for e in events
                )
                
                # Create a prompt that encourages conversational, thinking-aloud style
                style_prompts = {
                    "thoughtful": "Think out loud about what these changes might mean for the home and its occupants. Be conversational, as if you're sharing your thoughts with the homeowner.",
                    "analytical": "Analyze these changes and explain what they mean in a detailed, technical manner while still being conversational.",
                    "friendly": "Chat about these changes in a warm, friendly manner as if you're having a casual conversation with the homeowner.",
                    "concise": "Briefly comment on these changes in a conversational but efficient manner."
                }
                
                style_prompt = style_prompts.get(self.conversational_style, style_prompts["thoughtful"])
                
                prompt = (
                    f"You are an AI Overseer monitoring a smart home. The following state changes just occurred:\n\n"
                    f"{event_descriptions}\n\n"
                    f"{style_prompt} Focus on patterns, implications, and potential insights. "
                    f"Avoid simply repeating the state changes. Instead, interpret what they might mean."
                )
                
                try:
                    analysis = await self.agent.arun(prompt)
                    _LOGGER.info(f"Event Analysis: {analysis}")
                    
                    # Add to insights
                    self._add_insight(analysis, "event_analysis")
                    
                except Exception as e:
                    _LOGGER.error(f"Failed to analyze events: {str(e)}")

            except Exception as e:
                _LOGGER.error(f"Error in event processing loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    def _add_insight(self, content: str, source: str):
        """Add an insight to the history."""
        insight = InsightEntry(content, source)
        self.insights.append(insight)
        
        # Update the state entity
        self.hass.states.async_set(
            f"{DOMAIN}.insights",
            "active",
            {
                "last_insight": content,
                "source": source,
                "timestamp": insight.timestamp.isoformat(),
                "count": len(self.insights)
            }
        )
        
        # Notify subscribers
        self._notify_subscribers()
        
    def _notify_subscribers(self):
        """Notify websocket subscribers of new insights."""
        insights = list(self.insights)
        for connection in self.subscribers:
            connection.send_message(
                websocket_api.event_message(
                    None,
                    {"insights": [insight.as_dict() for insight in insights]}
                )
            )
                
    async def process_conversation_query(self, text: str) -> str:
        """Process a conversation query and return a response."""
        try:
            # Enhance the query with context about what the agent can do
            enhanced_query = (
                f"User query: {text}\n\n"
                "You are an AI Overseer for a smart home. Respond conversationally to the user's query "
                "about the state of their home. Use the tools available to you to get information about "
                "the current state of devices if needed."
            )
            
            response = await self.agent.arun(enhanced_query)
            return response
        except Exception as e:
            _LOGGER.error(f"Error processing conversation query: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

class OverseerConversationAgent:
    """Conversation agent for the Overseer Agent."""
    
    def __init__(self, overseer_agent: "OverseerAgent"):
        """Initialize the conversation agent."""
        self.overseer_agent = overseer_agent
        self.hass = overseer_agent.hass
        
    @property
    def attribution(self) -> Dict[str, Any]:
        """Return attribution."""
        return {
            "name": "AI Overseer Agent",
            "icon": "mdi:robot-outline",
        }
    
    # For newer versions of Home Assistant
    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a user input and return a response."""
        response_text = await self.overseer_agent.process_conversation_query(user_input.text)
        
        # Add to insights
        self.overseer_agent._add_insight(
            f"User asked: {user_input.text}\nMy response: {response_text}",
            "conversation"
        )
        
        return ConversationResult(
            response=response_text,
            conversation_id=user_input.conversation_id,
        )
    
    # For older versions of Home Assistant
    async def async_converse(
        self, text: str, conversation_id: Optional[str] = None, context: Optional[Context] = None
    ) -> ConversationResult:
        """Process a user input and return a response."""
        response_text = await self.overseer_agent.process_conversation_query(text)
        
        # Add to insights
        self.overseer_agent._add_insight(
            f"User asked: {text}\nMy response: {response_text}",
            "conversation"
        )
        
        return ConversationResult(
            response=response_text,
            conversation_id=conversation_id,
        )
    
    # For compatibility with all versions
    async def async_register(self) -> None:
        """Register this agent."""
        try:
            # Try newer method first
            conversation.async_register_agent(self.hass, self)
            _LOGGER.info("Registered conversation agent using new API")
        except (AttributeError, TypeError):
            try:
                # Fall back to older method
                await conversation.async_register(self.hass, self)
                _LOGGER.info("Registered conversation agent using legacy API")
            except Exception as e:
                _LOGGER.error(f"Failed to register conversation agent: {str(e)}")
                
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Overseer Agent component."""
    conf = config.get(DOMAIN, {})
    
    # Initialize the overseer agent
    agent = OverseerAgent(hass, conf)
    hass.data[DOMAIN] = agent
    
    # Register the frontend custom card
    await _register_frontend_resources(hass)
    
    # Start the agent
    return await agent.async_start()

async def _register_frontend_resources(hass: HomeAssistant) -> None:
    """Register frontend resources for the custom card."""
    try:
        # Copy the JS files to the www directory
        www_dir = hass.config.path("www")
        overseer_dir = os.path.join(www_dir, "overseer-agent")
        
        if not os.path.exists(overseer_dir):
            await hass.async_add_executor_job(os.makedirs, overseer_dir, exist_ok=True)
            
        # Get the path to the JS files in our component
        component_dir = os.path.dirname(os.path.abspath(__file__))
        js_files = [
            ("www/overseer-card.js", "overseer-card.js"),
            ("www/card-loader.js", "card-loader.js")
        ]
        
        # Copy the files using async executor to avoid blocking
        for src_rel_path, dest_filename in js_files:
            src_path = os.path.join(component_dir, src_rel_path)
            dest_path = os.path.join(overseer_dir, dest_filename)
            
            if os.path.exists(src_path) and not os.path.exists(dest_path):
                await hass.async_add_executor_job(
                    shutil.copy2, src_path, dest_path
                )
                _LOGGER.info(f"Copied {dest_filename} to {dest_path}")
        
        # Register the resources
        try:
            # For newer Home Assistant versions
            resources = ResourceStorageCollection(hass, "lovelace")
        except TypeError:
            # For older Home Assistant versions
            try:
                from homeassistant.components.lovelace import ResourceStorageCollection as OldResourceStorageCollection
                resources = OldResourceStorageCollection(hass)
            except (ImportError, TypeError):
                _LOGGER.error("Failed to initialize ResourceStorageCollection. Lovelace resources will not be registered.")
                return
                
        await resources.async_get_info()
        
        # Check if our resources are already registered
        resource_urls = [
            "/local/overseer-agent/overseer-card.js",
            "/local/overseer-agent/card-loader.js"
        ]
        
        for resource_url in resource_urls:
            resource_exists = False
            for resource in resources.async_items():
                if resource["url"] == resource_url:
                    _LOGGER.info(f"Resource {resource_url} already registered")
                    resource_exists = True
                    break
                    
            if not resource_exists:
                try:
                    await resources.async_create_item({
                        "res_type": "module",
                        "url": resource_url
                    })
                    _LOGGER.info(f"Registered resource {resource_url}")
                except Exception as e:
                    _LOGGER.error(f"Failed to register resource {resource_url}: {str(e)}")
            
        # Schedule a reload of lovelace to pick up the new resources
        async def reload_lovelace(_now=None):
            """Reload lovelace to pick up the new resources."""
            await hass.services.async_call("lovelace", "reload_resources", {})
            _LOGGER.info("Reloaded Lovelace resources")
            
        async_call_later(hass, 10, reload_lovelace)
            
    except Exception as e:
        _LOGGER.error(f"Failed to register frontend resources: {str(e)}")
