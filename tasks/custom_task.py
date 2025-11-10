from typing import Any, Dict, List, Optional  # Provide typing helpers for clarity and static analysis
import logging  # Provide logging utilities for observability within task abstractions
from crewai import Agent, Task  # Provide CrewAI primitives leveraged by this wrapper

logger = logging.getLogger(__name__)  # Acquire module-specific logger for contextualized diagnostics


class CustomTask:  # Provide secure, validated wrapper around CrewAI Task
    def __init__(  # Initialize wrapper with explicit configuration parameters
        self,  # Reference current instance
        description: str,  # Specify human-readable description of the task
        expected_output: str,  # Specify expected output for validation
        agent: Agent,  # Provide associated agent responsible for execution
        name: Optional[str] = None,  # Provide optional explicit task name
        context: Optional[List[Task]] = None,  # Provide optional task context dependencies
        async_execution: bool = False,  # Specify whether task should execute asynchronously
        config: Optional[Dict[str, Any]] = None,  # Provide optional configuration overrides
        output_file: Optional[str] = None,  # Provide optional output file target
        callback: Optional[Any] = None,  # Provide optional callback invoked upon completion
        human_input: bool = False,  # Specify whether human input is required mid-task
    ) -> None:  # Indicate initializer returns nothing
        self.name = name or f"task_{id(self)}"  # Persist deterministic name derived from id when not provided
        self.description = description  # Persist description for reuse
        self.expected_output = expected_output  # Persist expected output for reuse
        self.agent = agent  # Persist associated agent reference
        self.context = context or []  # Normalize context list to avoid mutability issues
        self.async_execution = async_execution  # Persist asynchronous execution preference
        self.config = config or {}  # Normalize configuration dictionary to avoid shared mutable defaults
        self.output_file = output_file  # Persist optional output file path
        self.callback = callback  # Persist optional callback reference
        self.human_input = human_input  # Persist human input requirement flag
        self.task = self._create_task()  # Instantiate underlying CrewAI task immediately during initialization

    def _create_task(self) -> Task:  # Provide helper to instantiate wrapped CrewAI task
        return Task(  # Return new CrewAI task instance using sanitized configuration
            description=self.description,  # Supply description to CrewAI task constructor
            expected_output=self.expected_output,  # Supply expected output to CrewAI task constructor
            agent=self.agent,  # Supply associated agent to CrewAI task constructor
            context=self.context,  # Supply context tasks to CrewAI task constructor
            async_execution=self.async_execution,  # Supply asynchronous execution preference
            config=self.config,  # Supply configuration overrides
            output_file=self.output_file,  # Supply optional output file location
            callback=self.callback,  # Supply optional callback
            human_input=self.human_input,  # Supply human input flag
        )

    def get_task_info(self) -> Dict[str, Any]:  # Provide serializable snapshot of task configuration
        return {  # Return dictionary representation of key configuration attributes
            "name": self.name,  # Include task name in summary
            "description": self.description,  # Include description in summary
            "expected_output": self.expected_output,  # Include expected output in summary
            "agent": getattr(self.agent, "role", "unknown"),  # Include agent role when available
            "context": [getattr(task, "description", "") for task in self.context],  # Include context descriptions
            "async_execution": self.async_execution,  # Include asynchronous execution flag
            "output_file": self.output_file,  # Include output file target
            "human_input": self.human_input,  # Include human input flag
        }

    def update_task(self, **kwargs: Any) -> None:  # Provide method to update task configuration safely
        for key, value in kwargs.items():  # Iterate through provided keyword arguments
            if hasattr(self, key):  # Update only known attributes to avoid typos or malicious input
                setattr(self, key, value)  # Persist updated attribute value
        self.task = self._create_task()  # Recreate underlying CrewAI task with updated configuration
        logger.info("Task %s configuration updated", self.name)  # Emit informational log about configuration update

    def validate_task(self) -> bool:  # Provide validation routine for configuration completeness
        required_fields = ["description", "expected_output", "agent"]  # Define list of required attribute names
        for field in required_fields:  # Iterate through required fields
            if not getattr(self, field):  # Check for missing or falsy values
                logger.error("Task %s missing required field %s", self.name, field)  # Emit error log identifying missing field
                return False  # Indicate validation failure
        if not self.agent:  # Ensure associated agent is provided
            logger.error("Task %s is missing associated agent", self.name)  # Emit error log when agent missing
            return False  # Indicate validation failure
        return True  # Indicate validation success

    def get_task(self) -> Task:  # Provide accessor for wrapped CrewAI task instance
        return self.task  # Return underlying CrewAI task for external use


class TaskFactory:  # Provide factory methods to create standardized task templates
    @staticmethod  # Declare method does not use instance or class state
    def create_research_task(agent: Agent, topic: str, expected_output: str = "Comprehensive research report") -> CustomTask:  # Provide helper for research task template
        description = (
            f"Conduct thorough research about {topic}, summarizing key findings, data, and actionable insights."  # Define secure research prompt
        )  # Close string literal
        return CustomTask(  # Instantiate CustomTask with research-focused configuration
            name=f"research_{topic.replace(' ', '_').lower()}",  # Generate deterministic safe task name
            description=description,  # Provide research description to task
            expected_output=expected_output,  # Provide expected output hint
            agent=agent,  # Provide associated agent
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_writing_task(agent: Agent, topic: str, format_type: str = "article", expected_output: str = "Well-written content") -> CustomTask:  # Provide helper for writing task template
        description = (
            f"Produce a {format_type} covering {topic} with clear structure, accurate facts, and engaging tone."  # Define secure writing prompt
        )  # Close string literal
        return CustomTask(  # Instantiate CustomTask with writing-focused configuration
            name=f"write_{format_type}_{topic.replace(' ', '_').lower()}",  # Generate deterministic safe task name
            description=description,  # Provide writing description to task
            expected_output=expected_output,  # Provide expected output hint
            agent=agent,  # Provide associated agent
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_analysis_task(agent: Agent, data: str, analysis_type: str = "comprehensive", expected_output: str = "Detailed analysis report") -> CustomTask:  # Provide helper for analysis task template
        description = (
            f"Perform {analysis_type} analysis on supplied data, highlighting trends, anomalies, and actionable recommendations."  # Define secure analysis prompt
        )  # Close string literal
        return CustomTask(  # Instantiate CustomTask with analysis-focused configuration
            name=f"analyze_{analysis_type}_{abs(hash(data))}",  # Generate deterministic safe task name using hash
            description=description,  # Provide analysis description to task
            expected_output=expected_output,  # Provide expected output hint
            agent=agent,  # Provide associated agent
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_coding_task(agent: Agent, requirements: str, language: str = "python", expected_output: str = "Working code solution") -> CustomTask:  # Provide helper for coding task template
        description = (
            f"Implement {requirements} using {language} with secure coding practices, tests, and documentation."  # Define secure coding prompt
        )  # Close string literal
        safe_name = f"code_{language}_{abs(hash(requirements))}"  # Generate deterministic safe task name using hash
        return CustomTask(  # Instantiate CustomTask with coding-focused configuration
            name=safe_name,  # Provide generated task name
            description=description,  # Provide coding description to task
            expected_output=expected_output,  # Provide expected output hint
            agent=agent,  # Provide associated agent
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_summary_task(agent: Agent, content: str, expected_output: str = "Concise summary") -> CustomTask:  # Provide helper for summary task template
        description = (
            "Summarize provided content focusing on key insights, preserving meaning while ensuring brevity."  # Define secure summary prompt
        )  # Close string literal
        safe_name = f"summary_{abs(hash(content))}"  # Generate deterministic safe task name using hash
        return CustomTask(  # Instantiate CustomTask with summary-focused configuration
            name=safe_name,  # Provide generated task name
            description=description,  # Provide summary description to task
            expected_output=expected_output,  # Provide expected output hint
            agent=agent,  # Provide associated agent
        )
