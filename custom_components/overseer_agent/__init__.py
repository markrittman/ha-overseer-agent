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
from pydantic import Field

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
    
    # Import tool creation functions
    try:
        from langchain.tools import Tool
        _LOGGER.info("Using langchain.tools.Tool")
    except ImportError:
        from langchain_core.tools import Tool
        _LOGGER.info("Using langchain_core.tools.Tool")
        
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
        from langchain.tools import BaseTool, Tool
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
        Tool = None
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

def create_state_summary_tool(state_history: Dict[str, Any]) -> Tool:
    """Create a state summary tool using the Tool factory function."""
    
    def _run_tool(query: str) -> str:
        """Summarize the current state."""
        summaries = []
        for entity_id, state_data in state_history.items():
            summaries.append(f"{entity_id}: {state_data['state']}")
        return "\n".join(summaries)
    
    async def _arun_tool(query: str) -> str:
        """Run async version."""
        return _run_tool(query)
    
    return Tool(
        name="state_summary",
        description="Summarize the current state of the smart home system and its devices",
        func=_run_tool,
        coroutine=_arun_tool,
    )


def create_entity_query_tool(state_history: Dict[str, Any], hass: HomeAssistant) -> Tool:
    """Create an entity query tool using the Tool factory function."""
    
    def _run_tool(query: str) -> str:
        """Get detailed information about specific entities or domains."""
        # Extract entity_id or domain from query
        entity_id = query.strip().lower()
        
        # Check if it's a specific entity
        if entity_id in state_history:
            state_data = state_history[entity_id]
            return f"Entity: {entity_id}\nState: {state_data['state']}\nAttributes: {state_data['attributes']}"
        
        # Check if it's a domain
        domain_entities = {}
        for eid, state_data in state_history.items():
            if eid.startswith(entity_id + "."):
                domain_entities[eid] = state_data
        
        if domain_entities:
            result = [f"Domain: {entity_id}"]
            for eid, state_data in domain_entities.items():
                result.append(f"  {eid}: {state_data['state']}")
            return "\n".join(result)
        
        return f"No entity or domain found matching '{entity_id}'"
    
    async def _arun_tool(query: str) -> str:
        """Run async version."""
        return _run_tool(query)
    
    return Tool(
        name="entity_query",
        description="Get detailed information about specific entities or domains in the smart home",
        func=_run_tool,
        coroutine=_arun_tool,
    )

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
            try:
                # Check if credentials file exists
                import os
                credentials_path = self.config["google_cloud_credentials"]
                if not os.path.exists(credentials_path):
                    _LOGGER.error(f"Google Cloud credentials file not found at: {credentials_path}")
                    _LOGGER.error("Please ensure your credentials file exists and the path is correct in configuration.yaml")
                    raise FileNotFoundError(f"Credentials file not found: {credentials_path}")
                
                # Check if project ID is provided
                project_id = self.config["google_cloud_project_id"]
                if not project_id:
                    _LOGGER.error("Google Cloud project ID is empty or not provided")
                    _LOGGER.error("Please set a valid project ID in configuration.yaml")
                    raise ValueError("Google Cloud project ID is required")
                
                # Set environment variable for credentials
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
                
                self.llm = VertexAI(
                    project=project_id,
                    credentials_path=credentials_path,
                    model_name="gemini-2.0-flash",
                    temperature=0.1,
                    max_output_tokens=1024
                )
                _LOGGER.info(f"Successfully initialized VertexAI with project ID: {project_id}")
            except FileNotFoundError as e:
                _LOGGER.error(f"Google Cloud credentials file not found: {str(e)}")
                raise
            except ValueError as e:
                _LOGGER.error(f"Invalid configuration value: {str(e)}")
                raise
            except Exception as e:
                _LOGGER.error(f"Failed to initialize VertexAI: {str(e)}")
                import traceback
                _LOGGER.error(f"VertexAI initialization traceback: {traceback.format_exc()}")
                raise
            
            # Initialize tools
            tools = [
                create_state_summary_tool(self.state_history),
                create_entity_query_tool(self.state_history, self.hass)
            ]
            
            # Initialize the memory
            try:
                # Try newer memory API with ChatMessageHistory from langchain_core
                try:
                    from langchain_core.memory import ChatMessageHistory
                    from langchain_core.messages import HumanMessage, AIMessage
                    
                    # Create a chat message history
                    chat_memory = ChatMessageHistory()
                    
                    # Add a system message to initialize the conversation
                    system_message = "I am the Home Assistant AI Overseer Agent. I can help you understand what's happening in your home and answer questions about your devices."
                    chat_memory.add_message(HumanMessage(content=f"System: {system_message}"))
                    chat_memory.add_message(AIMessage(content="I'm ready to help you understand your home and devices."))
                    
                    # Create memory with proper parameters to avoid deprecation warnings
                    self.memory = ConversationBufferMemory(
                        chat_memory=chat_memory,
                        return_messages=True,
                        memory_key="chat_history",
                        output_key=None,  # Explicitly set to None to avoid deprecation warning
                        input_key=None    # Explicitly set to None to avoid deprecation warning
                    )
                    _LOGGER.info("Using newer memory API with ChatMessageHistory from langchain_core")
                except (ImportError, AttributeError) as e:
                    # Try older memory API but still with ChatMessageHistory
                    _LOGGER.debug(f"Could not use langchain_core memory: {str(e)}")
                    from langchain.memory import ChatMessageHistory
                    chat_memory = ChatMessageHistory()
                    self.memory = ConversationBufferMemory(
                        chat_memory=chat_memory,
                        return_messages=True,
                        memory_key="chat_history"
                    )
                    _LOGGER.info("Using older memory API with ChatMessageHistory from langchain")
            except Exception as e:
                # Fall back to basic memory API as a last resort
                _LOGGER.warning(f"Could not initialize memory with ChatMessageHistory: {str(e)}")
                self.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
                _LOGGER.info("Using basic memory API without ChatMessageHistory")
            
            # Initialize the agent
            # Note: LangChain agents are deprecated in favor of LangGraph.
            # We'll continue using agents for now, but plan to migrate to LangGraph in a future version.
            try:
                import warnings
                from langchain.agents import AgentExecutor
                
                # Temporarily suppress the LangChain deprecation warning
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
                    
                    # Create the agent with initialize_agent
                    self.agent = initialize_agent(
                        tools=tools,
                        llm=self.llm,
                        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                        memory=self.memory,
                        verbose=True
                    )
                    
                    # Log that we're aware of the deprecation
                    _LOGGER.info("Agent initialized successfully (using deprecated LangChain agents API)")
                    _LOGGER.info("Future versions will migrate to LangGraph for agent functionality")
            except Exception as e:
                _LOGGER.error(f"Failed to initialize agent: {str(e)}")
                import traceback
                _LOGGER.error(f"Agent initialization traceback: {traceback.format_exc()}")
                raise
            
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
            import traceback
            _LOGGER.error(f"Traceback: {traceback.format_exc()}")
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
        try:
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
        except Exception as e:
            _LOGGER.error(f"Failed to register services: {str(e)}")
            import traceback
            _LOGGER.error(f"Service registration traceback: {traceback.format_exc()}")
        
    def _register_websocket_commands(self):
        """Register websocket commands."""
        try:
            websocket_api.async_register_command(
                self.hass,
                WS_TYPE_OVERSEER_INSIGHTS,
                self._handle_websocket_insights,
                websocket_api.BASE_COMMAND_MESSAGE_SCHEMA.extend({
                    vol.Required("type"): WS_TYPE_OVERSEER_INSIGHTS,
                    vol.Optional("count"): vol.Coerce(int),
                })
            )
            
            # Use the same handler for subscription as it already handles adding to subscribers
            websocket_api.async_register_command(
                self.hass,
                WS_TYPE_OVERSEER_SUBSCRIBE,
                self._handle_websocket_insights,
                websocket_api.BASE_COMMAND_MESSAGE_SCHEMA.extend({
                    vol.Required("type"): WS_TYPE_OVERSEER_SUBSCRIBE,
                })
            )
            
            _LOGGER.info("Registered Overseer Agent websocket commands")
        except Exception as e:
            _LOGGER.error(f"Failed to register websocket commands: {str(e)}")
            import traceback
            _LOGGER.error(f"Websocket registration traceback: {traceback.format_exc()}")
            
    async def _handle_websocket_insights(self, hass, connection, msg):
        """Handle websocket requests for insights."""
        connection.send_result(msg["id"])
        
        # Add this connection to subscribers
        self.subscribers.add(connection)
        
        # Send current insights
        count = msg.get("count", MAX_INSIGHT_ITEMS)
        insights = list(self.insights)[-count:]
        
        connection.send_message(
            websocket_api.event_message(
                msg["id"],
                {"insights": [insight.as_dict() for insight in insights]}
            )
        )
        
        # Remove connection when it's closed
        @connection.async_remove
        async def unsub():
            self.subscribers.remove(connection)

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
        
        # Check if we should track this entity
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
        old_state = event.data.get("old_state")
        previous_state = old_state.state if old_state else None
        
        await self.event_queue.put({
            "type": "state_changed",
            "entity_id": entity_id,
            "state": new_state.state,
            "previous_state": previous_state,
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
                    f"Entity {e['entity_id']} changed from {e.get('previous_state', 'unknown')} to {e['state']}"
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
                    # Use async_add_executor_job to run the agent in a separate thread
                    # This prevents blocking the event loop
                    analysis = await self.hass.async_add_executor_job(
                        self.agent.run, prompt
                    )
                    _LOGGER.info(f"Event Analysis: {analysis}")
                    
                    # Add to insights
                    self._add_insight(
                        analysis, "event_analysis"
                    )
                    
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
                
    def process_conversation_query(self, text: str) -> str:
        """Process a conversation query and return a response."""
        try:
            # Enhance the query with context about what the agent can do
            enhanced_query = (
                f"User query: {text}\n\n"
                "You are an AI Overseer for a smart home. Respond conversationally to the user's query "
                "about the state of their home. Use the tools available to you to get information about "
                "the current state of devices if needed."
            )
            
            # Use run instead of arun to avoid blocking the event loop
            # This method is called from async_add_executor_job in _handle_query_service
            response = self.agent.run(enhanced_query)
            return response
        except Exception as e:
            _LOGGER.error(f"Error processing conversation query: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

    async def _handle_query_service(self, call):
        """Handle the query service call."""
        query = call.data.get("query")
        if not query:
            return
            
        try:
            response = await self.hass.async_add_executor_job(
                self.process_conversation_query, query
            )
            
            # Add insight
            self._add_insight(
                f"Query: {query}\nResponse: {response}",
                "service"
            )
        except Exception as e:
            _LOGGER.error(f"Error processing query: {str(e)}")
            
    async def _handle_analyze_entity_service(self, call):
        """Handle the analyze entity service call."""
        entity_id = call.data.get("entity_id")
        if not entity_id:
            return
            
        try:
            # Get entity state
            state = self.hass.states.get(entity_id)
            if not state:
                _LOGGER.warning(f"Entity {entity_id} not found")
                return
                
            # Create query about the entity
            query = f"Analyze the state and behavior of {entity_id}. Current state: {state.state}"
            
            # Process with LLM
            response = await self.hass.async_add_executor_job(
                self.process_conversation_query, query
            )
            
            # Add insight
            self._add_insight(
                f"Analysis of {entity_id}: {response}",
                "service"
            )
        except Exception as e:
            _LOGGER.error(f"Error analyzing entity: {str(e)}")
            
    async def _handle_analyze_domain_service(self, call):
        """Handle the analyze domain service call."""
        domain = call.data.get("domain")
        if not domain:
            return
            
        try:
            # Get all entities in the domain
            entities = []
            for entity_id in self.hass.states.async_entity_ids(domain):
                state = self.hass.states.get(entity_id)
                if state:
                    entities.append(f"{entity_id}: {state.state}")
                    
            if not entities:
                _LOGGER.warning(f"No entities found in domain {domain}")
                return
                
            # Create query about the domain
            entity_list = "\n".join(entities[:10])  # Limit to 10 entities to avoid token limits
            query = f"Analyze the state and behavior of the {domain} domain. Entities:\n{entity_list}"
            
            # Process with LLM
            response = await self.hass.async_add_executor_job(
                self.process_conversation_query, query
            )
            
            # Add insight
            self._add_insight(
                f"Analysis of {domain} domain: {response}",
                "service"
            )
        except Exception as e:
            _LOGGER.error(f"Error analyzing domain: {str(e)}")
            
    async def _handle_clear_insights_service(self, call):
        """Handle the clear insights service call."""
        try:
            # Clear insights
            self.insights.clear()
            
            # Update the state entity
            self.hass.states.async_set(
                f"{DOMAIN}.insights",
                "active",
                {
                    "last_insight": "",
                    "source": "system",
                    "timestamp": datetime.now().isoformat(),
                    "count": 0
                }
            )
            
            # Notify subscribers
            self._notify_subscribers()
            
            _LOGGER.info("Cleared all insights")
        except Exception as e:
            _LOGGER.error(f"Error clearing insights: {str(e)}")

class OverseerConversationAgent:
    """Conversation agent for the Overseer Agent."""
    
    def __init__(self, overseer_agent: "OverseerAgent"):
        """Initialize the conversation agent."""
        self.overseer_agent = overseer_agent
        self.hass = overseer_agent.hass
        
    @property
    def attribution(self):
        """Return attribution."""
        return {"name": "Overseer Agent", "icon": "mdi:robot-outline"}
        
    async def async_process(
            self, user_input: ConversationInput
        ) -> ConversationResult:
        """Process a user input and return a response."""
        text = user_input.text
        response = await self.hass.async_add_executor_job(
            self.overseer_agent.process_conversation_query, text
        )
        
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )
        
    async def async_converse(
            self, text: str, conversation_id: Optional[str] = None, context: Optional[Context] = None
        ) -> ConversationResult:
        """Process a user input and return a response."""
        response = await self.hass.async_add_executor_job(
            self.overseer_agent.process_conversation_query, text
        )
        
        return ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )
        
    async def async_register(self):
        """Register this agent."""
        try:
            # Try newer registration method first
            try:
                from homeassistant.components.conversation import async_register
                await async_register(self.hass, self)
                _LOGGER.info("Registered conversation agent using newer API")
                return
            except (ImportError, AttributeError) as e:
                _LOGGER.debug(f"Could not register with newer API: {str(e)}")
                
            # Try older registration method
            try:
                from homeassistant.components.conversation import async_register_agent
                async_register_agent(self.hass, self)
                _LOGGER.info("Registered conversation agent using older API")
                return
            except (ImportError, AttributeError) as e:
                _LOGGER.debug(f"Could not register with older API: {str(e)}")
                
            # Try even older method
            try:
                from homeassistant.components.conversation.agent import AbstractConversationAgent
                if isinstance(self, AbstractConversationAgent):
                    from homeassistant.components.conversation import async_set_agent
                    async_set_agent(self.hass, self)
                    _LOGGER.info("Registered conversation agent using legacy API")
                    return
            except (ImportError, AttributeError) as e:
                _LOGGER.debug(f"Could not register with legacy API: {str(e)}")
            
            # Try direct component registration as a last resort
            try:
                from homeassistant.components import conversation
                if hasattr(conversation, "DOMAIN"):
                    # Get the conversation component
                    component = self.hass.data.get(conversation.DOMAIN)
                    if component and hasattr(component, "async_set_agent"):
                        await component.async_set_agent(self)
                        _LOGGER.info("Registered conversation agent using component direct access")
                        return
            except Exception as e:
                _LOGGER.debug(f"Could not register with component direct access: {str(e)}")
                
            _LOGGER.error("Could not register conversation agent - conversation component API has changed")
            _LOGGER.error("Please check Home Assistant version compatibility or report this issue")
        except Exception as e:
            _LOGGER.error(f"Error registering conversation agent: {str(e)}")
            import traceback
            _LOGGER.error(f"Registration traceback: {traceback.format_exc()}")
                
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Overseer Agent component."""
    try:
        conf = config.get(DOMAIN, {})
        
        # Check if configuration exists
        if not conf:
            _LOGGER.error("No configuration found for overseer_agent in configuration.yaml")
            return False
            
        # Check for required configuration
        if "google_cloud_project_id" not in conf:
            _LOGGER.error("Missing required configuration: google_cloud_project_id")
            return False
            
        if "google_cloud_credentials" not in conf:
            _LOGGER.error("Missing required configuration: google_cloud_credentials")
            return False
        
        # Initialize the overseer agent
        agent = OverseerAgent(hass, conf)
        hass.data[DOMAIN] = agent
        
        # Register the frontend custom card
        await _register_frontend_resources(hass)
        
        # Start the agent
        return await agent.async_start()
    except Exception as e:
        _LOGGER.error(f"Failed to set up Overseer Agent component: {str(e)}")
        import traceback
        _LOGGER.error(f"Setup traceback: {traceback.format_exc()}")
        return False

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
