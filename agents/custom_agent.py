import logging  # Import logging to provide consistent diagnostic output
from typing import Any, Dict, List, Optional  # Import typing helpers to annotate data structures

from crewai import Agent  # Import CrewAI's Agent class to wrap underlying agent behavior
from langchain_community.llms import Ollama  # Import Ollama type to annotate large language model dependencies

logger = logging.getLogger(__name__)  # Configure a module-level logger for observability


class CustomAgent:  # Define a wrapper that manages CrewAI agents with additional metadata
    def __init__(  # Initialize the custom agent with configuration and instantiate the CrewAI agent
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        llm: Ollama,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int = 25,
        max_rpm: Optional[int] = None,
        cache: bool = True
    ) -> None:
        self.name = name  # Store the human-readable agent name
        self.role = role  # Store the functional role description
        self.goal = goal  # Store the agent's primary objective
        self.backstory = backstory  # Store contextual backstory to guide the agent
        self.llm = llm  # Store the large language model interface
        self.tools = tools or []  # Store resolved tool instances, defaulting to an empty list
        self.verbose = verbose  # Store whether to enable verbose logging for the agent
        self.allow_delegation = allow_delegation  # Store whether the agent may delegate tasks
        self.max_iter = max_iter  # Store the maximum reasoning iterations for the agent
        self.max_rpm = max_rpm  # Store the optional per-minute rate limit for the agent
        self.cache = cache  # Store whether CrewAI caching should be enabled
        self.agent = self._create_agent()  # Instantiate the underlying CrewAI agent based on the stored configuration

    def _create_agent(self) -> Agent:  # Internal helper that builds the CrewAI agent instance
        return Agent(  # Create the CrewAI agent using the stored attributes
            role=self.role,  # Supply the agent role
            goal=self.goal,  # Supply the agent goal
            backstory=self.backstory,  # Supply the agent backstory
            llm=self.llm,  # Supply the large language model interface
            tools=self.tools,  # Attach any configured tool instances
            verbose=self.verbose,  # Enable or disable verbose mode
            allow_delegation=self.allow_delegation,  # Enable or disable delegation
            max_iter=self.max_iter,  # Apply the reasoning iteration limit
            max_rpm=self.max_rpm,  # Apply the per-minute rate limit when provided
            cache=self.cache  # Apply the caching preference
        )

    def get_agent_info(self) -> Dict[str, Any]:  # Expose a serializable snapshot of the agent configuration
        return {  # Return a dictionary describing the agent's configuration
            "name": self.name,  # Include the agent name
            "role": self.role,  # Include the agent role
            "goal": self.goal,  # Include the agent goal
            "backstory": self.backstory,  # Include the agent backstory
            "tools": [getattr(tool, "name", str(tool)) for tool in self.tools],  # Include tool identifiers for observability
            "verbose": self.verbose,  # Include the verbose flag
            "allow_delegation": self.allow_delegation,  # Include the delegation flag
            "max_iter": self.max_iter,  # Include the reasoning iteration limit
            "max_rpm": self.max_rpm,  # Include the optional per-minute limit
            "cache": self.cache  # Include the caching preference
        }

    def update_agent(self, **kwargs: Any) -> None:  # Allow updating configuration attributes and rebuilding the underlying agent
        for key, value in kwargs.items():  # Iterate over the provided attribute overrides
            if hasattr(self, key):  # Ensure the attribute exists on the object
                setattr(self, key, value)  # Update the attribute with the new value
        self.agent = self._create_agent()  # Rebuild the underlying CrewAI agent with the updated configuration
        logger.info("Updated agent %s configuration", self.name)  # Log that the agent configuration was updated

    def validate_agent(self) -> bool:  # Validate that the agent has sufficient configuration to operate
        required_fields = ["role", "goal", "backstory", "llm"]  # Define mandatory attributes for a functional agent
        for field in required_fields:  # Iterate over the mandatory attributes
            if not getattr(self, field):  # Check whether the attribute is missing or empty
                logger.error("Agent %s missing required field %s", self.name, field)  # Log the missing requirement
                return False  # Indicate validation failure
        return True  # Indicate that the agent configuration is valid

    def get_agent(self) -> Agent:  # Provide direct access to the underlying CrewAI agent
        return self.agent  # Return the instantiated CrewAI agent


class AgentFactory:  # Provide convenience constructors for commonly used agents
    @staticmethod  # Indicate that the method does not depend on instance state
    def create_research_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Produce a research-focused agent
        return CustomAgent(  # Instantiate the custom agent with research-oriented defaults
            name="research_agent",  # Assign a descriptive name
            role="Research Specialist",  # Assign the research role
            goal="Conduct thorough research and provide accurate, comprehensive information",  # Set a research-focused goal
            backstory="Experienced researcher skilled at synthesizing accurate findings.",  # Provide a concise backstory
            llm=llm,  # Supply the shared language model interface
            verbose=verbose,  # Apply the caller's verbosity preference
            max_iter=15  # Reduce iteration count for efficiency in research tasks
        )

    @staticmethod  # Indicate that the method does not depend on instance state
    def create_writer_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Produce a writing-focused agent
        return CustomAgent(  # Instantiate the custom agent with writing-oriented defaults
            name="writer_agent",  # Assign a descriptive name
            role="Content Writer",  # Assign the writing role
            goal="Create engaging, well-structured content based on provided information",  # Set a writing-focused goal
            backstory="Skilled writer adept at transforming complex topics into clear narratives.",  # Provide a concise backstory
            llm=llm,  # Supply the shared language model interface
            verbose=verbose,  # Apply the caller's verbosity preference
            max_iter=10  # Limit iterations to encourage concise writing
        )

    @staticmethod  # Indicate that the method does not depend on instance state
    def create_analyst_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Produce an analytical agent
        return CustomAgent(  # Instantiate the custom agent with analytical defaults
            name="analyst_agent",  # Assign a descriptive name
            role="Data Analyst",  # Assign the analytical role
            goal="Analyze data and provide actionable insights and recommendations",  # Set an analysis-focused goal
            backstory="Data analyst who discovers trends and actionable recommendations from complex datasets.",  # Provide a concise backstory
            llm=llm,  # Supply the shared language model interface
            verbose=verbose,  # Apply the caller's verbosity preference
            max_iter=20  # Allow a moderate number of reasoning iterations for analysis depth
        )

    @staticmethod  # Indicate that the method does not depend on instance state
    def create_coder_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:  # Produce a coding-focused agent
        return CustomAgent(  # Instantiate the custom agent with coding defaults
            name="coder_agent",  # Assign a descriptive name
            role="Software Developer",  # Assign the development role
            goal="Write clean, efficient, and well-documented code",  # Set a coding-focused goal
            backstory="Seasoned software developer committed to sustainable, high quality engineering practices.",  # Provide a concise backstory
            llm=llm,  # Supply the shared language model interface
            verbose=verbose,  # Apply the caller's verbosity preference
            max_iter=25  # Allow ample reasoning iterations for complex coding tasks
        )
