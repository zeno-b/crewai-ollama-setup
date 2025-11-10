from __future__ import annotations  # Enable postponed evaluation of annotations for forward references

import logging  # Import logging to provide audit-friendly diagnostics
from typing import Any, Callable, Dict, List, Optional, Sequence  # Import typing helpers for clarity and linter support

from crewai import Agent, Task  # Import CrewAI primitives for task composition

logger = logging.getLogger(__name__)  # Initialize a module-level logger for structured logging


class CustomTask:  # Define a secure, validated wrapper around CrewAI Task objects
    """Custom task class with validation and structured metadata exposure."""  # Document the purpose of the task wrapper

    _ALLOWED_UPDATE_FIELDS = {  # Define allowed attributes that can be updated post-instantiation
        "name",
        "description",
        "expected_output",
        "agent",
        "context",
        "async_execution",
        "config",
        "output_file",
        "callback",
        "human_input",
    }  # Close the whitelist definition

    def __init__(  # Initialize the custom task wrapper
        self,
        description: str,  # Capture the task instructions
        expected_output: str,  # Capture the expected outcome description
        agent: Agent,  # Capture the agent responsible for executing the task
        name: Optional[str] = None,  # Allow overriding the auto-generated task name
        context: Optional[Sequence[Task]] = None,  # Allow providing dependent context tasks
        async_execution: bool = False,  # Toggle asynchronous execution
        config: Optional[Dict[str, Any]] = None,  # Allow passing task-specific configuration
        output_file: Optional[str] = None,  # Allow writing task results to a file
        callback: Optional[Callable[..., Any]] = None,  # Allow specifying a callback hook
        human_input: bool = False,  # Toggle whether the task expects human input
    ) -> None:  # Explicitly state the initializer returns nothing
        self.name = name or f"task_{id(self)}"  # Persist the task name, generating a unique default when none is supplied
        self.description = description  # Persist the task description
        self.expected_output = expected_output  # Persist the expected output description
        self.agent = agent  # Persist the assigned agent instance
        self.context = list(context or [])  # Persist a defensive copy of the context tasks
        self.async_execution = async_execution  # Persist the asynchronous execution flag
        self.config = dict(config or {})  # Persist a defensive copy of configuration options
        self.output_file = output_file  # Persist the optional output file path
        self.callback = callback  # Persist the optional callback reference
        self.human_input = human_input  # Persist the human-input flag
        self.task = self._create_task()  # Instantiate the underlying CrewAI task

    def _create_task(self) -> Task:  # Internal helper to build the CrewAI task
        return Task(  # Instantiate and return the CrewAI Task
            description=self.description,  # Supply the task description
            expected_output=self.expected_output,  # Supply the expected output description
            agent=self.agent,  # Supply the agent responsible for execution
            context=self.context,  # Supply the dependent context tasks
            async_execution=self.async_execution,  # Supply the asynchronous execution flag
            config=self.config,  # Supply the task-specific configuration
            output_file=self.output_file,  # Supply the optional output file path
            callback=self.callback,  # Supply the optional callback hook
            human_input=self.human_input,  # Supply the human-input flag
        )  # Finish constructing the CrewAI Task

    def get_task_info(self) -> Dict[str, Any]:  # Expose metadata describing the task
        return {  # Build and return a serializable dictionary
            "name": self.name,  # Include the task name
            "description": self.description,  # Include the task description
            "expected_output": self.expected_output,  # Include the expected output description
            "agent": getattr(self.agent, "role", str(self.agent)),  # Include agent role information without leaking internal state
            "context": [str(task.description) for task in self.context],  # Include string summaries of context tasks
            "async_execution": self.async_execution,  # Include the asynchronous execution flag
            "output_file": self.output_file,  # Include the optional output file path
            "human_input": self.human_input,  # Include the human-input flag
        }  # Close the metadata dictionary

    def update_task(self, **kwargs: Any) -> None:  # Allow controlled updates to task configuration
        for key, value in kwargs.items():  # Iterate over supplied updates
            if key not in self._ALLOWED_UPDATE_FIELDS:  # Reject unsupported updates
                logger.warning("Attempt to update unsupported task field %s", key)  # Log the unsafe update attempt
                continue  # Skip unsupported fields
            setattr(self, key, value)  # Apply the supported update
        self.task = self._create_task()  # Rebuild the underlying CrewAI task to reflect updates
        logger.info("Updated task configuration for %s", self.name)  # Log the successful update

    def validate_task(self) -> bool:  # Validate that the task is configured properly
        required_fields = ["description", "expected_output", "agent"]  # Define required attributes
        for field_name in required_fields:  # Iterate over required attributes
            if not getattr(self, field_name):  # Check for missing or falsy values
                logger.error("Missing required task field: %s", field_name)  # Log the validation failure
                return False  # Fail validation
        if self.agent is None:  # Ensure an agent is assigned
            logger.error("Task %s does not have an assigned agent", self.name)  # Log the missing agent
            return False  # Fail validation when agent is missing
        return True  # Return True when validation passes

    def get_task(self) -> Task:  # Provide access to the underlying CrewAI Task
        return self.task  # Return the cached task instance


class TaskFactory:  # Provide reusable helpers to construct common task patterns
    """Factory class for creating opinionated task definitions."""  # Document the purpose of the factory

    @staticmethod
    def create_research_task(  # Build a research-oriented task
        agent: Agent,  # Supply the agent responsible for research
        topic: str,  # Supply the research topic
        expected_output: str = "Comprehensive research report",  # Supply the task's expected output description
    ) -> CustomTask:  # Return a configured CustomTask wrapper
        description_lines = [  # Build the research task prompt line by line
            f"Conduct thorough research on the following topic: {topic}",  # Provide the primary instruction
            "",  # Insert a blank line for readability
            "Your research should include:",  # Introduce the research checklist
            "- Key findings and insights",  # Enumerate findings requirement
            "- Relevant statistics and data",  # Enumerate statistics requirement
            "- Current trends and developments",  # Enumerate trends requirement
            "- Expert opinions and sources",  # Enumerate expert perspective requirement
            "- Potential implications and future outlook",  # Enumerate forward-looking requirement
            "",  # Insert a blank line for readability
            "Ensure all information is accurate, up-to-date, and properly sourced.",  # Emphasize data quality requirements
        ]  # Close the research description lines
        description = "\n".join(description_lines)  # Join the lines into a multi-line string instruction
        return CustomTask(  # Instantiate the CustomTask wrapper using the assembled description
            name=f"research_{topic.replace(' ', '_').lower()}",  # Generate a deterministic task name
            description=description,  # Supply the constructed description
            expected_output=expected_output,  # Supply the expected output description
            agent=agent,  # Supply the responsible agent
        )  # Return the configured task

    @staticmethod
    def create_writing_task(  # Build a writing-oriented task
        agent: Agent,  # Supply the agent responsible for writing
        topic: str,  # Supply the writing topic
        format_type: str = "article",  # Supply the target content format
        expected_output: str = "Well-written content",  # Supply the task's expected output description
    ) -> CustomTask:  # Return a configured CustomTask wrapper
        description_lines = [  # Build the writing task prompt line by line
            f"Create a {format_type} about: {topic}",  # Provide the primary instruction
            "",  # Insert a blank line for readability
            "Requirements:",  # Introduce the writing checklist
            "- Engaging and informative content",  # Enumerate engagement requirement
            "- Clear structure with introduction, body, and conclusion",  # Enumerate structure requirement
            "- Appropriate tone for the target audience",  # Enumerate tone requirement
            "- Well-researched and factually accurate",  # Enumerate research requirement
            "- Proper grammar and style",  # Enumerate quality requirement
            "",  # Insert a blank line for readability
            "The content should be ready for publication.",  # Emphasize production readiness
        ]  # Close the writing description lines
        description = "\n".join(description_lines)  # Join the lines into a multi-line string instruction
        return CustomTask(  # Instantiate the CustomTask wrapper using the assembled description
            name=f"write_{format_type}_{topic.replace(' ', '_').lower()}",  # Generate a deterministic task name
            description=description,  # Supply the constructed description
            expected_output=expected_output,  # Supply the expected output description
            agent=agent,  # Supply the responsible agent
        )  # Return the configured task

    @staticmethod
    def create_analysis_task(  # Build an analysis-oriented task
        agent: Agent,  # Supply the agent responsible for analysis
        data: str,  # Supply the data description or identifier
        analysis_type: str = "comprehensive",  # Supply the analysis depth indicator
        expected_output: str = "Detailed analysis report",  # Supply the task's expected output description
    ) -> CustomTask:  # Return a configured CustomTask wrapper
        description_lines = [  # Build the analysis task prompt line by line
            f"Perform {analysis_type} analysis on the following data: {data}",  # Provide the primary instruction
            "",  # Insert a blank line for readability
            "Analysis requirements:",  # Introduce the analysis checklist
            "- Identify key patterns and trends",  # Enumerate pattern detection requirement
            "- Provide statistical insights",  # Enumerate statistical insight requirement
            "- Highlight anomalies or outliers",  # Enumerate anomaly detection requirement
            "- Offer actionable recommendations",  # Enumerate recommendation requirement
            "- Include visual descriptions where helpful",  # Enumerate visualization requirement
            "- Consider multiple perspectives",  # Enumerate perspective requirement
            "",  # Insert a blank line for readability
            "Deliver a comprehensive analysis that drives decision-making.",  # Emphasize decision-making impact
        ]  # Close the analysis description lines
        description = "\n".join(description_lines)  # Join the lines into a multi-line string instruction
        return CustomTask(  # Instantiate the CustomTask wrapper using the assembled description
            name=f"analyze_{analysis_type}_{abs(hash(data))}",  # Generate a deterministic task name using a hash to preserve privacy
            description=description,  # Supply the constructed description
            expected_output=expected_output,  # Supply the expected output description
            agent=agent,  # Supply the responsible agent
        )  # Return the configured task

    @staticmethod
    def create_coding_task(  # Build a coding-oriented task
        agent: Agent,  # Supply the agent responsible for coding
        requirements: str,  # Supply the coding requirements
        language: str = "python",  # Supply the implementation language
        expected_output: str = "Working code solution",  # Supply the task's expected output description
    ) -> CustomTask:  # Return a configured CustomTask wrapper
        description_lines = [  # Build the coding task prompt line by line
            f"Write {language} code to implement: {requirements}",  # Provide the primary instruction
            "",  # Insert a blank line for readability
            "Code requirements:",  # Introduce the coding checklist
            "- Clean, readable, and maintainable code",  # Enumerate readability requirement
            "- Proper error handling",  # Enumerate error-handling requirement
            "- Comprehensive comments and documentation",  # Enumerate documentation requirement
            "- Follow best practices and conventions",  # Enumerate standards requirement
            "- Include unit tests where appropriate",  # Enumerate testing requirement
            "- Optimize for performance",  # Enumerate performance requirement
            "",  # Insert a blank line for readability
            "Provide the complete, working code solution.",  # Emphasize completeness
        ]  # Close the coding description lines
        description = "\n".join(description_lines)  # Join the lines into a multi-line string instruction
        return CustomTask(  # Instantiate the CustomTask wrapper using the assembled description
            name=f"code_{language}_{requirements.replace(' ', '_').lower()[:20]}",  # Generate a deterministic, length-limited task name
            description=description,  # Supply the constructed description
            expected_output=expected_output,  # Supply the expected output description
            agent=agent,  # Supply the responsible agent
        )  # Return the configured task

    @staticmethod
    def create_summary_task(  # Build a summarization-oriented task
        agent: Agent,  # Supply the agent responsible for summarization
        content: str,  # Supply the content to summarize
        expected_output: str = "Concise summary",  # Supply the task's expected output description
    ) -> CustomTask:  # Return a configured CustomTask wrapper
        preview = content[:200] + ("..." if len(content) > 200 else "")  # Generate a preview snippet to avoid logging sensitive content
        description_lines = [  # Build the summary task prompt line by line
            f"Create a concise summary of the following content: {preview}",  # Provide the primary instruction
            "",  # Insert a blank line for readability
            "Summary requirements:",  # Introduce the summary checklist
            "- Capture the main points and key insights",  # Enumerate key point requirement
            "- Be concise yet comprehensive",  # Enumerate conciseness requirement
            "- Maintain the original meaning",  # Enumerate fidelity requirement
            "- Use clear and simple language",  # Enumerate clarity requirement
            "- Highlight the most important information",  # Enumerate importance requirement
            "",  # Insert a blank line for readability
            "The summary should be easily digestible and actionable.",  # Emphasize usability
        ]  # Close the summary description lines
        description = "\n".join(description_lines)  # Join the lines into a multi-line string instruction
        return CustomTask(  # Instantiate the CustomTask wrapper using the assembled description
            name=f"summary_{abs(hash(content))}",  # Generate a deterministic task name using a hash to preserve privacy
            description=description,  # Supply the constructed description
            expected_output=expected_output,  # Supply the expected output description
            agent=agent,  # Supply the responsible agent
        )  # Return the configured task
