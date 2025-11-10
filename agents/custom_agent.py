from __future__ import annotations  # Enable postponed evaluation of annotations for forward references

from typing import Any, Dict, List, Optional, Sequence  # Import typing helpers for clarity and linting support
import logging  # Import logging to capture diagnostic information securely

from crewai import Agent  # Import CrewAI's Agent to wrap custom agent configuration
from langchain_community.llms import Ollama  # Import the Ollama LLM client used by the agents

from crewai_tools import BaseTool  # Import BaseTool to type tools passed to the agent

logger = logging.getLogger(__name__)  # Initialize a module-level logger for structured logging


class CustomAgent:  # Define a security-conscious wrapper around CrewAI Agent instances
    """Custom agent class with enhanced validation and state tracking."""  # Document the purpose of the wrapper class

    _ALLOWED_UPDATE_FIELDS = {  # Define a whitelist of fields that can be updated post-instantiation
        "name",
        "role",
        "goal",
        "backstory",
        "llm",
        "tools",
        "verbose",
        "allow_delegation",
        "max_iter",
        "max_rpm",
        "cache",
    }  # Close the whitelist definition

    def __init__(  # Initialize the custom agent wrapper with validated settings
        self,
        name: str,  # Capture the agent's unique identifier
        role: str,  # Capture the agent's primary role description
        goal: str,  # Capture the agent's goal statement
        backstory: str,  # Capture contextual backstory for richer prompts
        llm: Ollama,  # Capture the Ollama language model instance to execute agent tasks
        tools: Optional[Sequence[BaseTool]] = None,  # Capture optional tool integrations for the agent
        verbose: bool = False,  # Toggle verbose execution logging
        allow_delegation: bool = False,  # Specify whether the agent can delegate tasks to others
        max_iter: int = 25,  # Limit the number of iterations to avoid runaway loops
        max_rpm: Optional[int] = None,  # Optionally limit requests per minute for rate control
        cache: bool = True,  # Toggle CrewAI caching behaviour to optimize repeated calls
    ) -> None:  # Explicitly state the initializer returns nothing
        self.name = name  # Persist the agent name as an instance attribute
        self.role = role  # Persist the agent role for reuse and inspection
        self.goal = goal  # Persist the agent goal for reference during execution
        self.backstory = backstory  # Persist the agent backstory to enrich prompts
        self.llm = llm  # Persist the Ollama model instance for later calls
        self.tools: List[Any] = list(tools or [])  # Store a defensive copy of provided tools while tolerating diverse tool inputs
        self.verbose = verbose  # Persist the verbosity preference
        self.allow_delegation = allow_delegation  # Persist the delegation flag
        self.max_iter = max_iter  # Persist the maximum iteration count
        self.max_rpm = max_rpm  # Persist the rate limit value
        self.cache = cache  # Persist the caching preference
        self.agent = self._create_agent()  # Instantiate the underlying CrewAI agent immediately

    def _create_agent(self) -> Agent:  # Internal helper to construct the CrewAI Agent from stored state
        return Agent(  # Create and return the CrewAI Agent instance
            role=self.role,  # Supply the agent role to CrewAI
            goal=self.goal,  # Supply the agent goal
            backstory=self.backstory,  # Supply the agent backstory for context
            llm=self.llm,  # Supply the Ollama model instance
            tools=self.tools or None,  # Supply tools only when at least one tool is configured
            verbose=self.verbose,  # Supply the verbosity preference
            allow_delegation=self.allow_delegation,  # Supply the delegation flag
            max_iter=self.max_iter,  # Supply the iteration limit
            max_rpm=self.max_rpm,  # Supply the rate limit if configured
            cache=self.cache,  # Supply the caching preference
        )  # Finish constructing the CrewAI Agent

    def get_agent_info(self) -> Dict[str, Any]:  # Expose a read-only view of the agent configuration
        return {  # Build and return a serializable dictionary of agent attributes
            "name": self.name,  # Include the agent name
            "role": self.role,  # Include the agent role
            "goal": self.goal,  # Include the agent goal
            "backstory": self.backstory,  # Include the agent backstory
            "tools": [getattr(tool, "name", str(tool)) for tool in self.tools],  # Include tool identifiers without assuming a specific tool type
            "verbose": self.verbose,  # Include the verbosity flag
            "allow_delegation": self.allow_delegation,  # Include the delegation flag
            "max_iter": self.max_iter,  # Include the iteration limit
            "max_rpm": self.max_rpm,  # Include the rate limit
            "cache": self.cache,  # Include the caching flag
        }  # Close the info dictionary

    def update_agent(self, **kwargs: Any) -> None:  # Allow controlled updates to the agent configuration
        for key, value in kwargs.items():  # Iterate over supplied updates
            if key not in self._ALLOWED_UPDATE_FIELDS:  # Reject attempts to update unsupported fields
                logger.warning("Attempt to update unsupported agent field %s", key)  # Log the unsafe update attempt
                continue  # Skip updating unsupported fields
            setattr(self, key, value)  # Apply the supported field update
        self.agent = self._create_agent()  # Rebuild the underlying CrewAI Agent to reflect new settings
        logger.info("Updated agent configuration for %s", self.name)  # Record the update for auditability

    def validate_agent(self) -> bool:  # Validate that the agent is configured correctly
        required_fields = ["role", "goal", "backstory"]  # Define fields that must be non-empty
        for field in required_fields:  # Iterate over required fields
            if not getattr(self, field):  # Check for missing or falsy values
                logger.error("Missing required agent field: %s", field)  # Log the configuration issue
                return False  # Fail validation on the first missing field
        if self.llm is None:  # Ensure the language model dependency is present
            logger.error("LLM instance is not configured for agent %s", self.name)  # Log the missing LLM
            return False  # Fail validation if LLM is absent
        return True  # Return True when all validation checks pass

    def get_agent(self) -> Agent:  # Provide access to the underlying CrewAI Agent
        return self.agent  # Return the cached CrewAI Agent instance


class AgentFactory:  # Expose convenience helpers for constructing well-known agent personas
    """Factory class for creating common agent configurations."""  # Document the factory purpose

    @staticmethod
    def create_research_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Build a research-focused agent wrapper
        return CustomAgent(  # Delegate to CustomAgent with preconfigured persona details
            name="research_agent",  # Supply a stable agent identifier
            role="Research Specialist",  # Describe the research role
            goal="Conduct thorough research and provide accurate, comprehensive information",  # Provide the goal statement
            backstory=(  # Provide a narrative backstory to guide the agent
                "You are an expert researcher with years of experience in gathering, analyzing, and synthesizing "
                "information from various sources. You excel at finding relevant data and presenting it in a clear, "
                "organized manner."  # Continue the descriptive backstory
            ),  # Close the backstory string
            llm=llm,  # Supply the language model dependency
            verbose=verbose,  # Propagate the verbosity preference
            max_iter=15,  # Set a tighter iteration cap tailored to research tasks
        )  # Return the newly constructed research agent

    @staticmethod
    def create_writer_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Build a writing-focused agent wrapper
        return CustomAgent(  # Delegate to CustomAgent with preconfigured persona details
            name="writer_agent",  # Supply a stable agent identifier
            role="Content Writer",  # Describe the writing role
            goal="Create engaging, well-structured content based on provided information",  # Provide the goal statement
            backstory=(  # Provide a narrative backstory to guide the agent
                "You are a skilled writer with expertise in creating compelling content across various formats. "
                "You excel at transforming complex information into clear, engaging narratives that resonate with "
                "the target audience."  # Continue the descriptive backstory
            ),  # Close the backstory string
            llm=llm,  # Supply the language model dependency
            verbose=verbose,  # Propagate the verbosity preference
            max_iter=10,  # Set a concise iteration cap tuned for writing tasks
        )  # Return the newly constructed writer agent

    @staticmethod
    def create_analyst_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Build an analysis-focused agent wrapper
        return CustomAgent(  # Delegate to CustomAgent with preconfigured persona details
            name="analyst_agent",  # Supply a stable agent identifier
            role="Data Analyst",  # Describe the analysis role
            goal="Analyze data and provide actionable insights and recommendations",  # Provide the goal statement
            backstory=(  # Provide a narrative backstory to guide the agent
                "You are a data analyst with strong analytical skills and the ability to extract meaningful insights "
                "from complex datasets. You excel at identifying patterns, trends, and providing data-driven "
                "recommendations."  # Continue the descriptive backstory
            ),  # Close the backstory string
            llm=llm,  # Supply the language model dependency
            verbose=verbose,  # Propagate the verbosity preference
            max_iter=20,  # Set an iteration cap suited for analytical workflows
        )  # Return the newly constructed analyst agent

    @staticmethod
    def create_coder_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Build a coding-focused agent wrapper
        return CustomAgent(  # Delegate to CustomAgent with preconfigured persona details
            name="coder_agent",  # Supply a stable agent identifier
            role="Software Developer",  # Describe the development role
            goal="Write clean, efficient, and well-documented code",  # Provide the goal statement
            backstory=(  # Provide a narrative backstory to guide the agent
                "You are an experienced software developer with expertise in multiple programming languages and "
                "frameworks. You excel at writing clean, maintainable code and solving complex technical challenges."  # Continue the descriptive backstory
            ),  # Close the backstory string
            llm=llm,  # Supply the language model dependency
            verbose=verbose,  # Propagate the verbosity preference
            max_iter=25,  # Set a generous iteration cap for coding tasks
        )  # Return the newly constructed coder agent
