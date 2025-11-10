import logging  # Import logging to emit structured diagnostic information
from typing import Any, Dict, List, Optional  # Import typing helpers to annotate task metadata

from crewai import Agent, Task  # Import CrewAI Agent and Task classes to orchestrate task execution

logger = logging.getLogger(__name__)  # Configure a module-level logger for task operations


class CustomTask:  # Define a wrapper that augments CrewAI tasks with metadata and validation
    def __init__(  # Initialize the custom task and instantiate the underlying CrewAI task
        self,
        description: str,
        expected_output: str,
        agent: Agent,
        name: Optional[str] = None,
        context: Optional[List[Task]] = None,
        async_execution: bool = False,
        config: Optional[Dict[str, Any]] = None,
        output_file: Optional[str] = None,
        callback: Optional[Any] = None,
        human_input: bool = False
    ) -> None:
        self.name = name or f"task_{id(self)}"  # Assign a stable task name, defaulting to a unique identifier
        self.description = description  # Store the task description that guides the agent
        self.expected_output = expected_output  # Store the expected output specification for validation
        self.agent = agent  # Store the agent responsible for executing the task
        self.context = context or []  # Store optional contextual tasks that inform execution
        self.async_execution = async_execution  # Store whether the task should execute asynchronously
        self.config = config or {}  # Store additional configuration parameters for CrewAI
        self.output_file = output_file  # Store an optional output file path for task results
        self.callback = callback  # Store an optional callback invoked after task completion
        self.human_input = human_input  # Store whether human input is required during execution
        self.task = self._create_task()  # Instantiate the underlying CrewAI task based on the stored configuration

    def _create_task(self) -> Task:  # Internal helper that builds a CrewAI task instance
        return Task(  # Create a CrewAI task using the stored attributes
            description=self.description,  # Supply the task description
            expected_output=self.expected_output,  # Supply the expected output specification
            agent=self.agent,  # Assign the agent that will execute the task
            context=self.context,  # Supply contextual tasks for reference
            async_execution=self.async_execution,  # Indicate whether the task should run asynchronously
            config=self.config,  # Supply additional configuration parameters
            output_file=self.output_file,  # Supply the optional output file path
            callback=self.callback,  # Supply the optional completion callback
            human_input=self.human_input  # Indicate whether human input is required
        )

    def get_task_info(self) -> Dict[str, Any]:  # Produce a serializable snapshot of task metadata
        return {  # Return a dictionary describing the task configuration
            "name": self.name,  # Include the task name
            "description": self.description,  # Include the task description
            "expected_output": self.expected_output,  # Include the expected output specification
            "agent": getattr(self.agent, "role", str(self.agent)),  # Include the agent role for readability
            "context": [task.description for task in self.context],  # Include descriptions of contextual tasks
            "async_execution": self.async_execution,  # Include whether the task runs asynchronously
            "output_file": self.output_file,  # Include the optional output file path
            "human_input": self.human_input  # Include whether human interaction is required
        }

    def update_task(self, **kwargs: Any) -> None:  # Allow updates to the task configuration and rebuild the task
        for key, value in kwargs.items():  # Iterate over the provided updates
            if hasattr(self, key):  # Ensure the attribute exists before updating
                setattr(self, key, value)  # Apply the update to the task attribute
        self.task = self._create_task()  # Rebuild the underlying CrewAI task with the updated configuration
        logger.info("Updated task %s configuration", self.name)  # Log that the task was updated

    def validate_task(self) -> bool:  # Validate that the task has the required configuration
        required_fields = ["description", "expected_output", "agent"]  # Define mandatory attributes for a task
        for field in required_fields:  # Iterate over each required attribute
            if not getattr(self, field):  # Check whether the attribute is missing or empty
                logger.error("Task %s missing required field %s", self.name, field)  # Log the missing requirement
                return False  # Indicate validation failure
        return True  # Indicate that the task configuration is valid

    def get_task(self) -> Task:  # Provide access to the underlying CrewAI task instance
        return self.task  # Return the instantiated CrewAI task


class TaskFactory:  # Provide convenience factory methods for common task types
    @staticmethod  # Indicate that the method does not rely on instance state
    def create_research_task(agent: Agent, topic: str, expected_output: str = "Comprehensive research report") -> CustomTask:  # Build a research-oriented task
        description = (  # Construct a concise task description for research activities
            f"Conduct thorough research on the topic: {topic}. Summarize key findings, trends, and authoritative sources."  # Provide explicit research instructions
        )  # Close the description construction
        return CustomTask(  # Instantiate the custom task with research defaults
            name=f"research_{topic.replace(' ', '_').lower()}",  # Create a deterministic task name derived from the topic
            description=description,  # Supply the research description
            expected_output=expected_output,  # Supply the expected research output
            agent=agent  # Assign the agent that will execute the research
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_writing_task(agent: Agent, topic: str, format_type: str = "article", expected_output: str = "Well-written content") -> CustomTask:  # Build a writing task
        description = (  # Build the writing task description
            f"Draft a {format_type} covering {topic}. Ensure clear structure, factual accuracy, and engaging storytelling."  # Provide explicit writing guidance
        )  # Close the description construction
        return CustomTask(  # Instantiate the custom task with writing defaults
            name=f"write_{format_type}_{topic.replace(' ', '_').lower()}",  # Create a deterministic task name based on the topic and format
            description=description,  # Supply the writing description
            expected_output=expected_output,  # Supply the expected writing output
            agent=agent  # Assign the agent that will produce the content
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_analysis_task(agent: Agent, data: str, analysis_type: str = "comprehensive", expected_output: str = "Detailed analysis report") -> CustomTask:  # Build an analysis task
        description = (  # Build the analytical task description
            f"Perform a {analysis_type} analysis on the provided data: {data}. Highlight insights, anomalies, and actionable recommendations."  # Provide analysis instructions
        )  # Close the description construction
        return CustomTask(  # Instantiate the custom task with analysis defaults
            name=f"analyze_{analysis_type}_{abs(hash(data))}",  # Create a deterministic task name using a hash of the data for stability
            description=description,  # Supply the analysis description
            expected_output=expected_output,  # Supply the expected analysis output
            agent=agent  # Assign the agent that will conduct the analysis
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_coding_task(agent: Agent, requirements: str, language: str = "python", expected_output: str = "Working code solution") -> CustomTask:  # Build a coding task
        description = (  # Build the coding task description
            f"Implement the following requirements using {language}: {requirements}. Deliver clean, tested, and well-documented code."  # Provide coding instructions
        )  # Close the description construction
        return CustomTask(  # Instantiate the custom task with coding defaults
            name=f"code_{language}_{abs(hash(requirements))}",  # Create a deterministic task name using a hash of the requirements
            description=description,  # Supply the coding description
            expected_output=expected_output,  # Supply the expected coding output
            agent=agent  # Assign the agent that will write the code
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_summary_task(agent: Agent, content: str, expected_output: str = "Concise summary") -> CustomTask:  # Build a summarization task
        description = (  # Build the summary task description
            f"Summarize the following content into a concise, actionable overview: {content[:200]}..."  # Provide summarization instructions with context preview
        )  # Close the description construction
        return CustomTask(  # Instantiate the custom task with summary defaults
            name=f"summary_{abs(hash(content))}",  # Create a deterministic task name using a hash of the content
            description=description,  # Supply the summary description
            expected_output=expected_output,  # Supply the expected summary output
            agent=agent  # Assign the agent that will craft the summary
        )
