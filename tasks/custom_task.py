from typing import List, Optional, Dict, Any
from crewai import Task
from crewai import Agent
import logging

logger = logging.getLogger(__name__)

class CustomTask:
    """Custom task class with enhanced functionality"""
    
    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Agent,
        name: Optional[str] = None,
        context: Optional[List[Task]] = None,
        async_execution: bool = False,
        config: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None,
        callback: Optional[callable] = None,
        human_input: bool = False
    ):
        self.name = name or f"task_{id(self)}"
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context or []
        self.async_execution = async_execution
        self.config = config or {}
        self.output_file = output_file
        self.callback = callback
        self.human_input = human_input
        
        # Create the actual CrewAI task
        self.task = self._create_task()
    
    def _create_task(self) -> Task:
        """Create the underlying CrewAI task"""
        return Task(
            description=self.description,
            expected_output=self.expected_output,
            agent=self.agent,
            context=self.context,
            async_execution=self.async_execution,
            config=self.config,
            output_file=self.output_file,
            callback=self.callback,
            human_input=self.human_input
        )
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get task information"""
        return {
            "name": self.name,
            "description": self.description,
            "expected_output": self.expected_output,
            "agent": str(self.agent.role),
            "context": [str(task.description) for task in self.context],
            "async_execution": self.async_execution,
            "output_file": self.output_file,
            "human_input": self.human_input
        }
    
    def update_task(self, **kwargs) -> None:
        """Update task properties"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recreate the task with updated properties
        self.task = self._create_task()
        logger.info(f"Updated task: {self.name}")
    
    def validate_task(self) -> bool:
        """Validate task configuration"""
        required_fields = ["description", "expected_output", "agent"]
        
        for field in required_fields:
            if not getattr(self, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        if not self.agent:
            logger.error("Agent not provided")
            return False
        
        return True
    
    def get_task(self) -> Task:
        """Get the underlying CrewAI task"""
        return self.task

class TaskFactory:
    """Factory class for creating common tasks"""
    
    @staticmethod
    def create_research_task(
        agent: Agent,
        topic: str,
        expected_output: str = "Comprehensive research report"
    ) -> CustomTask:
        """Create a research task"""
        description = f"""
        Conduct thorough research on the following topic: {topic}
        
        Your research should include:
        - Key findings and insights
        - Relevant statistics and data
        - Current trends and developments
        - Expert opinions and sources
        - Potential implications and future outlook
        
        Ensure all information is accurate, up-to-date, and properly sourced.
        """
        
        return CustomTask(
            name=f"research_{topic.replace(' ', '_').lower()}",
            description=description,
            expected_output=expected_output,
            agent=agent
        )
    
    @staticmethod
    def create_writing_task(
        agent: Agent,
        topic: str,
        format_type: str = "article",
        expected_output: str = "Well-written content"
    ) -> CustomTask:
        """Create a writing task"""
        description = f"""
        Create a {format_type} about: {topic}
        
        Requirements:
        - Engaging and informative content
        - Clear structure with introduction, body, and conclusion
        - Appropriate tone for the target audience
        - Well-researched and factually accurate
        - Proper grammar and style
        
        The content should be ready for publication.
        """
        
        return CustomTask(
            name=f"write_{format_type}_{topic.replace(' ', '_').lower()}",
            description=description,
            expected_output=expected_output,
            agent=agent
        )
    
    @staticmethod
    def create_analysis_task(
        agent: Agent,
        data: str,
        analysis_type: str = "comprehensive",
        expected_output: str = "Detailed analysis report"
    ) -> CustomTask:
        """Create an analysis task"""
        description = f"""
        Perform {analysis_type} analysis on the following data: {data}
        
        Analysis requirements:
        - Identify key patterns and trends
        - Provide statistical insights
        - Highlight anomalies or outliers
        - Offer actionable recommendations
        - Include visual descriptions where helpful
        - Consider multiple perspectives
        
        Deliver a comprehensive analysis that drives decision-making.
        """
        
        return CustomTask(
            name=f"analyze_{analysis_type}_{id(data)}",
            description=description,
            expected_output=expected_output,
            agent=agent
        )
    
    @staticmethod
    def create_coding_task(
        agent: Agent,
        requirements: str,
        language: str = "python",
        expected_output: str = "Working code solution"
    ) -> CustomTask:
        """Create a coding task"""
        description = f"""
        Write {language} code to implement: {requirements}
        
        Code requirements:
        - Clean, readable, and maintainable code
        - Proper error handling
        - Comprehensive comments and documentation
        - Follow best practices and conventions
        - Include unit tests where appropriate
        - Optimize for performance
        
        Provide the complete, working code solution.
        """
        
        return CustomTask(
            name=f"code_{language}_{requirements.replace(' ', '_').lower()[:20]}",
            description=description,
            expected_output=expected_output,
            agent=agent
        )
    
    @staticmethod
    def create_summary_task(
        agent: Agent,
        content: str,
        expected_output: str = "Concise summary"
    ) -> CustomTask:
        """Create a summary task"""
        description = f"""
        Create a concise summary of the following content: {content[:200]}...
        
        Summary requirements:
        - Capture the main points and key insights
        - Be concise yet comprehensive
        - Maintain the original meaning
        - Use clear and simple language
        - Highlight the most important information
        
        The summary should be easily digestible and actionable.
        """
        
        return CustomTask(
            name=f"summary_{id(content)}",
            description=description,
            expected_output=expected_output,
            agent=agent
        )
