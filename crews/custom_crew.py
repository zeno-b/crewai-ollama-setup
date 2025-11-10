from typing import List, Optional, Dict, Any
from crewai import Crew, Task, Agent
from langchain_community.llms import Ollama
import logging
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..agents.custom_agent import CustomAgent, AgentFactory
from ..tasks.custom_task import CustomTask, TaskFactory
from ..config.settings import settings

logger = logging.getLogger(__name__)

class CustomCrew:
    """Custom crew class with enhanced functionality"""
    
    def __init__(
        self,
        name: str,
        agents: List[CustomAgent],
        tasks: List[CustomTask],
        verbose: bool = False,
        process: str = "sequential",
        manager_llm: Optional[Ollama] = None,
        function_calling_llm: Optional[Ollama] = None,
        config: Optional[Dict[str, Any]] = None,
        cache: bool = True,
        max_rpm: Optional[int] = None,
        share_crew: bool = False,
        output_log_file: Optional[str] = None,
        embedder_config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.process = process
        self.manager_llm = manager_llm
        self.function_calling_llm = function_calling_llm
        self.config = config or {}
        self.cache = cache
        self.max_rpm = max_rpm
        self.share_crew = share_crew
        self.output_log_file = output_log_file
        self.embedder_config = embedder_config
        
        # Create the actual CrewAI crew
        self.crew = self._create_crew()
        
        # Execution tracking
        self.execution_history = []
        self.current_execution = None
    
    def _create_crew(self) -> Crew:
        """Create the underlying CrewAI crew"""
        return Crew(
            agents=[agent.get_agent() for agent in self.agents],
            tasks=[task.get_task() for task in self.tasks],
            verbose=self.verbose,
            process=self.process,
            manager_llm=self.manager_llm,
            function_calling_llm=self.function_calling_llm,
            config=self.config,
            cache=self.cache,
            max_rpm=self.max_rpm,
            share_crew=self.share_crew,
            output_log_file=self.output_log_file,
            embedder_config=self.embedder_config
        )
    
    def get_crew_info(self) -> Dict[str, Any]:
        """Get crew information"""
        return {
            "name": self.name,
            "agents": [agent.get_agent_info() for agent in self.agents],
            "tasks": [task.get_task_info() for task in self.tasks],
            "process": self.process,
            "verbose": self.verbose,
            "cache": self.cache,
            "max_rpm": self.max_rpm,
            "execution_history": self.execution_history
        }
    
    def add_agent(self, agent: CustomAgent) -> None:
        """Add a new agent to the crew"""
        self.agents.append(agent)
        self.crew = self._create_crew()
        logger.info(f"Added agent {agent.name} to crew {self.name}")
    
    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent from the crew"""
        initial_count = len(self.agents)
        self.agents = [agent for agent in self.agents if agent.name != agent_name]
        
        if len(self.agents) < initial_count:
            self.crew = self._create_crew()
            logger.info(f"Removed agent {agent_name} from crew {self.name}")
            return True
        
        logger.warning(f"Agent {agent_name} not found in crew {self.name}")
        return False
    
    def add_task(self, task: CustomTask) -> None:
        """Add a new task to the crew"""
        self.tasks.append(task)
        self.crew = self._create_crew()
        logger.info(f"Added task {task.name} to crew {self.name}")
    
    def remove_task(self, task_name: str) -> bool:
        """Remove a task from the crew"""
        initial_count = len(self.tasks)
        self.tasks = [task for task in self.tasks if task.name != task_name]
        
        if len(self.tasks) < initial_count:
            self.crew = self._create_crew()
            logger.info(f"Removed task {task_name} from crew {self.name}")
            return True
        
        logger.warning(f"Task {task_name} not found in crew {self.name}")
        return False
    
    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the crew"""
        execution_id = f"{self.name}_{datetime.now().isoformat()}"
        
        try:
            logger.info(f"Starting crew execution: {execution_id}")
            
            execution_start = datetime.now()
            result = self.crew.kickoff(inputs=inputs)
            execution_end = datetime.now()
            
            execution_record = {
                "execution_id": execution_id,
                "start_time": execution_start.isoformat(),
                "end_time": execution_end.isoformat(),
                "duration": (execution_end - execution_start).total_seconds(),
                "inputs": inputs,
                "result": str(result),
                "status": "success"
            }
            
            self.execution_history.append(execution_record)
            logger.info(f"Crew execution completed: {execution_id}")
            
            return {
                "execution_id": execution_id,
                "result": result,
                "duration": execution_record["duration"],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Crew execution failed: {execution_id} - {str(e)}")
            
            execution_record = {
                "execution_id": execution_id,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": 0,
                "inputs": inputs,
                "error": str(e),
                "status": "failed"
            }
            
            self.execution_history.append(execution_record)
            
            return {
                "execution_id": execution_id,
                "error": str(e),
                "status": "failed"
            }
    
    async def execute_async(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the crew asynchronously"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.execute,
                inputs
            )
        
        return result
    
    def validate_crew(self) -> bool:
        """Validate crew configuration"""
        if not self.agents:
            logger.error("No agents configured")
            return False
        
        if not self.tasks:
            logger.error("No tasks configured")
            return False
        
        # Validate all agents
        for agent in self.agents:
            if not agent.validate_agent():
                logger.error(f"Invalid agent: {agent.name}")
                return False
        
        # Validate all tasks
        for task in self.tasks:
            if not task.validate_task():
                logger.error(f"Invalid task: {task.name}")
                return False
        
        return True
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
    def export_execution_history(self, file_path: str) -> bool:
        """Export execution history to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.execution_history, f, indent=2)
            logger.info(f"Execution history exported to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export execution history: {str(e)}")
            return False

class CrewFactory:
    """Factory class for creating common crews"""
    
    @staticmethod
    def create_research_crew(
        llm: Ollama,
        topic: str,
        verbose: bool = False
    ) -> CustomCrew:
        """Create a research crew"""
        research_agent = AgentFactory.create_research_agent(llm, verbose)
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)
        
        research_task = TaskFactory.create_research_task(
            research_agent.get_agent(),
            topic
        )
        writing_task = TaskFactory.create_writing_task(
            writer_agent.get_agent(),
            topic,
            "research report"
        )
        
        return CustomCrew(
            name=f"research_crew_{topic.replace(' ', '_').lower()}",
            agents=[research_agent, writer_agent],
            tasks=[research_task, writing_task],
            verbose=verbose,
            process="sequential"
        )
    
    @staticmethod
    def create_analysis_crew(
        llm: Ollama,
        data: str,
        verbose: bool = False
    ) -> CustomCrew:
        """Create an analysis crew"""
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)
        
        analysis_task = TaskFactory.create_analysis_task(
            analyst_agent.get_agent(),
            data
        )
        summary_task = TaskFactory.create_summary_task(
            writer_agent.get_agent(),
            "analysis results"
        )
        
        return CustomCrew(
            name=f"analysis_crew_{id(data)}",
            agents=[analyst_agent, writer_agent],
            tasks=[analysis_task, summary_task],
            verbose=verbose,
            process="sequential"
        )
    
    @staticmethod
    def create_coding_crew(
        llm: Ollama,
        requirements: str,
        language: str = "python",
        verbose: bool = False
    ) -> CustomCrew:
        """Create a coding crew"""
        coder_agent = AgentFactory.create_coder_agent(llm, verbose)
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)
        
        coding_task = TaskFactory.create_coding_task(
            coder_agent.get_agent(),
            requirements,
            language
        )
        review_task = TaskFactory.create_analysis_task(
            analyst_agent.get_agent(),
            "code review"
        )
        
        return CustomCrew(
            name=f"coding_crew_{requirements.replace(' ', '_').lower()[:20]}",
            agents=[coder_agent, analyst_agent],
            tasks=[coding_task, review_task],
            verbose=verbose,
            process="sequential"
        )
