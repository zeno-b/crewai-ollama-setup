from typing import Any, Dict, List, Optional  # Provide typing helpers for clarity and static analysis
import logging  # Provide logging utilities for observability within agent abstractions
from crewai import Agent  # Provide CrewAI Agent primitive leveraged by this wrapper
from langchain_community.llms import Ollama  # Provide Ollama large language model type for type safety

logger = logging.getLogger(__name__)  # Acquire module-specific logger for contextualized diagnostics


class CustomAgent:  # Provide secure, validated wrapper around CrewAI Agent
    def __init__(  # Initialize wrapper with explicit configuration parameters
        self,  # Reference current instance
        name: str,  # Specify unique agent name for tracking
        role: str,  # Specify agent role for CrewAI context
        goal: str,  # Specify agent goal for AI reasoning
        backstory: str,  # Specify descriptive backstory for prompt grounding
        llm: Ollama,  # Provide initialized Ollama client instance
        tools: Optional[List[str]] = None,  # Provide optional list of tool identifiers
        verbose: bool = False,  # Control verbosity for debugging or auditing
        allow_delegation: bool = False,  # Specify delegation capability flag
        max_iter: int = 25,  # Specify maximum reasoning iterations
        max_rpm: Optional[int] = None,  # Specify optional rate limit per minute
        cache: bool = True,  # Control caching strategy for responses
    ) -> None:  # Indicate initializer returns nothing
        self.name = name  # Persist agent name for later reference
        self.role = role  # Persist agent role for later reference
        self.goal = goal  # Persist agent goal for later reference
        self.backstory = backstory  # Persist agent backstory for later reference
        self.llm = llm  # Persist Ollama client for reuse
        self.tools = tools or []  # Normalize tools list to avoid mutability issues
        self.verbose = verbose  # Persist verbosity preference
        self.allow_delegation = allow_delegation  # Persist delegation capability flag
        self.max_iter = max_iter  # Persist iteration cap for deterministic behavior
        self.max_rpm = max_rpm  # Persist optional rate limit for throttling
        self.cache = cache  # Persist cache preference
        self.agent = self._create_agent()  # Instantiate underlying CrewAI agent immediately during initialization

    def _create_agent(self) -> Agent:  # Provide helper to instantiate wrapped CrewAI agent
        return Agent(  # Return new CrewAI agent instance using sanitized configuration
            role=self.role,  # Supply role to CrewAI agent constructor
            goal=self.goal,  # Supply goal to CrewAI agent constructor
            backstory=self.backstory,  # Supply backstory to CrewAI agent constructor
            llm=self.llm,  # Supply Ollama client to CrewAI agent constructor
            verbose=self.verbose,  # Supply verbosity preference to CrewAI agent constructor
            allow_delegation=self.allow_delegation,  # Supply delegation capability to CrewAI agent constructor
            max_iter=self.max_iter,  # Supply iteration cap to CrewAI agent constructor
            max_rpm=self.max_rpm,  # Supply optional rate limit to CrewAI agent constructor
            cache=self.cache,  # Supply cache preference to CrewAI agent constructor
        )

    def get_agent_info(self) -> Dict[str, Any]:  # Provide serializable snapshot of agent configuration
        return {  # Return dictionary representation of key configuration attributes
            "name": self.name,  # Include agent name in summary
            "role": self.role,  # Include agent role in summary
            "goal": self.goal,  # Include agent goal in summary
            "backstory": self.backstory,  # Include agent backstory in summary
            "tools": self.tools,  # Include tools assigned to agent
            "verbose": self.verbose,  # Include verbosity preference
            "allow_delegation": self.allow_delegation,  # Include delegation capability
            "max_iter": self.max_iter,  # Include iteration cap
            "max_rpm": self.max_rpm,  # Include rate limit if set
            "cache": self.cache,  # Include cache preference
        }

    def update_agent(self, **kwargs: Any) -> None:  # Provide method to update agent configuration safely
        for key, value in kwargs.items():  # Iterate through provided keyword arguments
            if hasattr(self, key):  # Update only known attributes to avoid typos or malicious input
                setattr(self, key, value)  # Persist updated attribute value
        self.agent = self._create_agent()  # Recreate underlying CrewAI agent with updated configuration
        logger.info("Agent %s configuration updated", self.name)  # Emit informational log about configuration update

    def validate_agent(self) -> bool:  # Provide validation routine for configuration completeness
        required_fields = ["role", "goal", "backstory"]  # Define list of required attribute names
        for field in required_fields:  # Iterate through required fields
            if not getattr(self, field):  # Check for missing or falsy values
                logger.error("Agent %s missing required field %s", self.name, field)  # Emit error log identifying missing field
                return False  # Indicate validation failure
        if not self.llm:  # Ensure language model client is provided
            logger.error("Agent %s is missing LLM configuration", self.name)  # Emit error log when LLM missing
            return False  # Indicate validation failure
        return True  # Indicate validation success

    def get_agent(self) -> Agent:  # Provide accessor for wrapped CrewAI agent instance
        return self.agent  # Return underlying CrewAI agent for external use


class AgentFactory:  # Provide factory methods to create predefined agent profiles
    @staticmethod  # Declare method does not use instance or class state
    def create_research_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Provide helper for research agent profile
        return CustomAgent(  # Instantiate CustomAgent with research-focused configuration
            name="research_agent",  # Assign deterministic agent name
            role="Research Specialist",  # Assign role describing expertise
            goal="Conduct thorough research and provide accurate, comprehensive information",  # Assign agent goal statement
            backstory=(
                "Expert researcher skilled in gathering, analyzing, and synthesizing complex information."  # Provide concise backstory aligned with security guidelines
            ),  # Close string literal
            llm=llm,  # Provide shared Ollama client instance
            verbose=verbose,  # Propagate verbosity preference from caller
            max_iter=15,  # Tune iteration count for research tasks
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_writer_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Provide helper for writer agent profile
        return CustomAgent(  # Instantiate CustomAgent with writer-focused configuration
            name="writer_agent",  # Assign deterministic agent name
            role="Content Writer",  # Assign role describing expertise
            goal="Create engaging content derived from researched information",  # Assign agent goal statement
            backstory="Skilled writer adept at translating complex topics into accessible narratives.",  # Provide concise backstory supportive of security policies
            llm=llm,  # Provide shared Ollama client instance
            verbose=verbose,  # Propagate verbosity preference from caller
            max_iter=10,  # Tune iteration count for writing tasks
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_analyst_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Provide helper for analyst agent profile
        return CustomAgent(  # Instantiate CustomAgent with analyst-focused configuration
            name="analyst_agent",  # Assign deterministic agent name
            role="Data Analyst",  # Assign role describing expertise
            goal="Analyze data to extract actionable insights and recommendations",  # Assign agent goal statement
            backstory="Analytical thinker experienced in interpreting data trends and presenting findings clearly.",  # Provide concise backstory aligned with security best practices
            llm=llm,  # Provide shared Ollama client instance
            verbose=verbose,  # Propagate verbosity preference from caller
            max_iter=20,  # Tune iteration count for analytical tasks
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_coder_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Provide helper for coding agent profile
        return CustomAgent(  # Instantiate CustomAgent with coding-focused configuration
            name="coder_agent",  # Assign deterministic agent name
            role="Software Developer",  # Assign role describing expertise
            goal="Design secure, maintainable, and efficient software solutions",  # Assign agent goal statement
            backstory="Seasoned engineer focused on writing clean, secure, and performant code.",  # Provide concise backstory reflecting secure coding emphasis
            llm=llm,  # Provide shared Ollama client instance
            verbose=verbose,  # Propagate verbosity preference from caller
            max_iter=25,  # Tune iteration count for development tasks
        )
