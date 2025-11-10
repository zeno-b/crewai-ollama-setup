from typing import List, Optional, Dict, Any
from crewai import Agent
from langchain_community.llms import Ollama
import logging

logger = logging.getLogger(__name__)

class CustomAgent:
    """Custom agent class with enhanced functionality"""
    
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        llm: Ollama,
        tools: Optional[List[str]] = None,
        verbose: bool = False,
        allow_delegation: bool = False,
        max_iter: int = 25,
        max_rpm: Optional[int] = None,
        cache: bool = True
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.tools = tools or []
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.cache = cache
        
        # Create the actual CrewAI agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying CrewAI agent"""
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            llm=self.llm,
            verbose=self.verbose,
            allow_delegation=self.allow_delegation,
            max_iter=self.max_iter,
            max_rpm=self.max_rpm,
            cache=self.cache
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "verbose": self.verbose,
            "allow_delegation": self.allow_delegation,
            "max_iter": self.max_iter,
            "max_rpm": self.max_rpm,
            "cache": self.cache
        }
    
    def update_agent(self, **kwargs) -> None:
        """Update agent properties"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recreate the agent with updated properties
        self.agent = self._create_agent()
        logger.info(f"Updated agent: {self.name}")
    
    def validate_agent(self) -> bool:
        """Validate agent configuration"""
        required_fields = ["role", "goal", "backstory"]
        
        for field in required_fields:
            if not getattr(self, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        if not self.llm:
            logger.error("LLM not configured")
            return False
        
        return True
    
    def get_agent(self) -> Agent:
        """Get the underlying CrewAI agent"""
        return self.agent

class AgentFactory:
    """Factory class for creating agents"""
    
    @staticmethod
    def create_research_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:
        """Create a research agent"""
        return CustomAgent(
            name="research_agent",
            role="Research Specialist",
            goal="Conduct thorough research and provide accurate, comprehensive information",
            backstory="""You are an expert researcher with years of experience in gathering,
            analyzing, and synthesizing information from various sources. You excel at finding
            relevant data and presenting it in a clear, organized manner.""",
            llm=llm,
            verbose=verbose,
            max_iter=15
        )
    
    @staticmethod
    def create_writer_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:
        """Create a writer agent"""
        return CustomAgent(
            name="writer_agent",
            role="Content Writer",
            goal="Create engaging, well-structured content based on provided information",
            backstory="""You are a skilled writer with expertise in creating compelling content
            across various formats. You excel at transforming complex information into clear,
            engaging narratives that resonate with the target audience.""",
            llm=llm,
            verbose=verbose,
            max_iter=10
        )
    
    @staticmethod
    def create_analyst_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:
        """Create an analyst agent"""
        return CustomAgent(
            name="analyst_agent",
            role="Data Analyst",
            goal="Analyze data and provide actionable insights and recommendations",
            backstory="""You are a data analyst with strong analytical skills and the ability to
            extract meaningful insights from complex datasets. You excel at identifying patterns,
            trends, and providing data-driven recommendations.""",
            llm=llm,
            verbose=verbose,
            max_iter=20
        )
    
    @staticmethod
    def create_coder_agent(llm: Ollama, verbose: bool = False) -> CustomAgent:
        """Create a coder agent"""
        return CustomAgent(
            name="coder_agent",
            role="Software Developer",
            goal="Write clean, efficient, and well-documented code",
            backstory="""You are an experienced software developer with expertise in multiple
            programming languages and frameworks. You excel at writing clean, maintainable code
            and solving complex technical challenges.""",
            llm=llm,
            verbose=verbose,
            max_iter=25
        )
