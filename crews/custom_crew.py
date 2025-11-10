from __future__ import annotations  # Enable postponed evaluation of annotations for forward references

import asyncio  # Import asyncio to support asynchronous crew execution
import json  # Import json to serialize execution history safely
import logging  # Import logging for structured observability
from datetime import datetime  # Import datetime for timestamping execution events
from pathlib import Path  # Import Path to perform secure filesystem operations
from typing import Any, Dict, List, Optional  # Import typing helpers for clarity and quality checks

from crewai import Crew  # Import the CrewAI primitive used to compose crews
from langchain_community.llms import Ollama  # Import the Ollama LLM type for type hints

from agents.custom_agent import AgentFactory, CustomAgent  # Import custom agent helpers using absolute paths for stability
from config.settings import settings  # Import runtime settings to enforce operational policies
from tasks.custom_task import CustomTask, TaskFactory  # Import task helpers using absolute paths for stability

logger = logging.getLogger(__name__)  # Initialize a module-level logger for audit-friendly logging


class CustomCrew:  # Define a secure and observable wrapper around CrewAI crews
    """Custom crew class with validation, observability, and safe file operations."""  # Document the purpose of the crew wrapper

    def __init__(  # Initialize the crew wrapper with explicit parameters
        self,
        name: str,  # Capture the crew identifier for logging and storage
        agents: List[CustomAgent],  # Capture the participating agents
        tasks: List[CustomTask],  # Capture the ordered list of tasks
        verbose: bool = False,  # Toggle verbose CrewAI logging
        process: str = "sequential",  # Choose the task execution strategy
        manager_llm: Optional[Ollama] = None,  # Optionally supply a manager-level LLM
        function_calling_llm: Optional[Ollama] = None,  # Optionally supply an LLM dedicated to function calling
        config: Optional[Dict[str, Any]] = None,  # Allow passing additional CrewAI configuration
        cache: bool = True,  # Control CrewAI caching behaviour
        max_rpm: Optional[int] = None,  # Limit requests per minute to stay within rate caps
        share_crew: bool = False,  # Toggle CrewAI crew sharing features
        output_log_file: Optional[str] = None,  # Optionally direct CrewAI to write execution logs to a file
        embedder_config: Optional[Dict[str, Any]] = None,  # Optionally supply embedding configuration
    ) -> None:  # Explicitly state the initializer returns nothing
        if len(agents) > settings.max_agents:  # Enforce a maximum agent count for resource control
            raise ValueError("Agent count exceeds configured maximum.")  # Fail fast to maintain compliance
        if len(tasks) > settings.max_tasks:  # Enforce a maximum task count for resource control
            raise ValueError("Task count exceeds configured maximum.")  # Fail fast to maintain compliance
        self.name = name  # Persist the crew identifier
        self.agents = agents  # Persist the list of agents
        self.tasks = tasks  # Persist the list of tasks
        self.verbose = verbose  # Persist the verbosity preference
        self.process = process  # Persist the execution strategy
        self.manager_llm = manager_llm  # Persist the manager LLM reference
        self.function_calling_llm = function_calling_llm  # Persist the function-calling LLM reference
        self.config = config or {}  # Persist additional CrewAI configuration with a safe default
        self.cache = cache  # Persist the caching preference
        self.max_rpm = max_rpm  # Persist the rate limit setting
        self.share_crew = share_crew  # Persist the sharing option
        self.output_log_file = output_log_file  # Persist the optional output log file path
        self.embedder_config = embedder_config  # Persist the embedder configuration
        self.crew = self._create_crew()  # Construct the underlying CrewAI crew immediately
        self.execution_history: List[Dict[str, Any]] = []  # Initialize an in-memory execution ledger

    def _create_crew(self) -> Crew:  # Internal helper to construct the CrewAI crew object
        return Crew(  # Instantiate and return the CrewAI crew
            agents=[agent.get_agent() for agent in self.agents],  # Extract CrewAI agents from the wrappers
            tasks=[task.get_task() for task in self.tasks],  # Extract CrewAI tasks from the wrappers
            verbose=self.verbose,  # Supply the verbosity preference
            process=self.process,  # Supply the execution strategy
            manager_llm=self.manager_llm,  # Supply the manager LLM if configured
            function_calling_llm=self.function_calling_llm,  # Supply the function-calling LLM if configured
            config=self.config,  # Supply additional configuration options
            cache=self.cache,  # Supply the caching preference
            max_rpm=self.max_rpm,  # Supply any rate limit
            share_crew=self.share_crew,  # Supply the sharing preference
            output_log_file=self.output_log_file,  # Supply the optional output log file path
            embedder_config=self.embedder_config,  # Supply embedding configuration if provided
        )  # Finish constructing the CrewAI crew

    def get_crew_info(self) -> Dict[str, Any]:  # Provide a serializable snapshot of crew configuration and history
        return {  # Build and return a dictionary describing the crew
            "name": self.name,  # Include the crew name
            "agents": [agent.get_agent_info() for agent in self.agents],  # Include details for each agent
            "tasks": [task.get_task_info() for task in self.tasks],  # Include details for each task
            "process": self.process,  # Include the execution strategy
            "verbose": self.verbose,  # Include the verbosity flag
            "cache": self.cache,  # Include the caching flag
            "max_rpm": self.max_rpm,  # Include the rate limit
            "execution_history": list(self.execution_history),  # Include a shallow copy of the execution ledger
        }  # Close the info dictionary

    def add_agent(self, agent: CustomAgent) -> None:  # Append a new agent while respecting limits
        if len(self.agents) >= settings.max_agents:  # Guard against exceeding the configured limit
            raise ValueError("Cannot add agent: maximum agent count reached.")  # Fail fast to maintain compliance
        self.agents.append(agent)  # Append the new agent
        self.crew = self._create_crew()  # Rebuild the CrewAI crew to include the new agent
        logger.info("Added agent %s to crew %s", agent.name, self.name)  # Log the update for auditing

    def remove_agent(self, agent_name: str) -> bool:  # Remove the agent with the supplied name
        initial_count = len(self.agents)  # Capture the original agent count
        self.agents = [agent for agent in self.agents if agent.name != agent_name]  # Filter out the matching agent
        if len(self.agents) < initial_count:  # Detect whether an agent was removed
            self.crew = self._create_crew()  # Rebuild the crew to reflect the updated membership
            logger.info("Removed agent %s from crew %s", agent_name, self.name)  # Log the successful removal
            return True  # Indicate a successful removal
        logger.warning("Agent %s not found in crew %s", agent_name, self.name)  # Log the failed removal attempt
        return False  # Indicate that no agent matched the supplied name

    def add_task(self, task: CustomTask) -> None:  # Append a new task while respecting limits
        if len(self.tasks) >= settings.max_tasks:  # Guard against exceeding the configured limit
            raise ValueError("Cannot add task: maximum task count reached.")  # Fail fast to maintain compliance
        self.tasks.append(task)  # Append the new task
        self.crew = self._create_crew()  # Rebuild the CrewAI crew to include the new task
        logger.info("Added task %s to crew %s", task.name, self.name)  # Log the update for auditing

    def remove_task(self, task_name: str) -> bool:  # Remove the task with the supplied name
        initial_count = len(self.tasks)  # Capture the original task count
        self.tasks = [task for task in self.tasks if task.name != task_name]  # Filter out the matching task
        if len(self.tasks) < initial_count:  # Detect whether a task was removed
            self.crew = self._create_crew()  # Rebuild the crew to reflect the updated task list
            logger.info("Removed task %s from crew %s", task_name, self.name)  # Log the successful removal
            return True  # Indicate a successful removal
        logger.warning("Task %s not found in crew %s", task_name, self.name)  # Log the failed removal attempt
        return False  # Indicate that no task matched the supplied name

    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Execute the crew synchronously
        execution_id = f"{self.name}_{datetime.utcnow().isoformat()}"  # Generate a unique execution identifier
        try:  # Wrap execution in a try block to capture failures
            logger.info("Starting crew execution %s", execution_id)  # Log the execution start event
            execution_start = datetime.utcnow()  # Record the start time
            result = self.crew.kickoff(inputs=inputs)  # Execute the crew workflow
            execution_end = datetime.utcnow()  # Record the end time
            execution_record = {  # Build a structured execution record
                "execution_id": execution_id,  # Include the execution identifier
                "start_time": execution_start.isoformat(),  # Include the ISO start timestamp
                "end_time": execution_end.isoformat(),  # Include the ISO end timestamp
                "duration": (execution_end - execution_start).total_seconds(),  # Include the duration in seconds
                "inputs": inputs,  # Include the provided inputs for traceability
                "result": str(result),  # Include the serialized result
                "status": "success",  # Mark the execution as successful
            }  # Close the execution record
            self.execution_history.append(execution_record)  # Append the record to the execution ledger
            logger.info("Crew execution %s completed successfully", execution_id)  # Log the completion
            return {  # Return a concise response payload
                "execution_id": execution_id,  # Provide the execution identifier
                "result": result,  # Provide the raw result
                "duration": execution_record["duration"],  # Provide the execution duration
                "status": "success",  # Provide the status flag
            }  # Close the response payload
        except Exception as error:  # Capture any execution failures
            logger.exception("Crew execution %s failed", execution_id)  # Log the exception with stack trace
            execution_record = {  # Build a structured failure record
                "execution_id": execution_id,  # Include the execution identifier
                "start_time": datetime.utcnow().isoformat(),  # Record the time the failure was handled
                "end_time": datetime.utcnow().isoformat(),  # Record the completion timestamp
                "duration": 0,  # Set duration to zero for failures
                "inputs": inputs,  # Include the provided inputs for traceability
                "error": str(error),  # Include the error message
                "status": "failed",  # Mark the execution as failed
            }  # Close the failure record
            self.execution_history.append(execution_record)  # Append the failure record to the ledger
            return {  # Return a concise error payload
                "execution_id": execution_id,  # Provide the execution identifier
                "error": str(error),  # Provide the error message
                "status": "failed",  # Provide the failure status
            }  # Close the error payload

    async def execute_async(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Execute the crew asynchronously
        return await asyncio.to_thread(self.execute, inputs)  # Offload the synchronous execute call to a worker thread

    def validate_crew(self) -> bool:  # Validate that the crew configuration is complete
        if not self.agents:  # Ensure at least one agent is configured
            logger.error("Crew %s has no agents configured", self.name)  # Log the validation failure
            return False  # Fail validation when agents are missing
        if not self.tasks:  # Ensure at least one task is configured
            logger.error("Crew %s has no tasks configured", self.name)  # Log the validation failure
            return False  # Fail validation when tasks are missing
        for agent in self.agents:  # Iterate over agents to validate each one
            if not agent.validate_agent():  # Invoke agent-specific validation logic
                logger.error("Agent %s failed validation for crew %s", agent.name, self.name)  # Log the failed validation
                return False  # Fail validation on the first invalid agent
        for task in self.tasks:  # Iterate over tasks to validate each one
            if not task.validate_task():  # Invoke task-specific validation logic
                logger.error("Task %s failed validation for crew %s", task.name, self.name)  # Log the failed validation
                return False  # Fail validation on the first invalid task
        return True  # Return True when all validation checks pass

    def get_execution_history(self) -> List[Dict[str, Any]]:  # Expose the execution history
        return list(self.execution_history)  # Return a shallow copy to avoid external mutation

    def export_execution_history(self, file_path: str) -> bool:  # Persist execution history to disk securely
        try:  # Wrap file operations in a try block to catch IO errors
            target_path = Path(file_path).resolve()  # Resolve the destination path to prevent directory traversal
            workspace_root = Path.cwd().resolve()  # Resolve the workspace root directory
            if not str(target_path).startswith(str(workspace_root)):  # Ensure the export stays within the workspace
                raise ValueError("Export path is outside the permitted workspace directory.")  # Reject unsafe paths
            target_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists
            target_path.write_text(json.dumps(self.execution_history, indent=2), encoding="utf-8")  # Write the execution history as formatted JSON
            logger.info("Execution history exported to %s", target_path)  # Log the successful export
            return True  # Indicate success
        except Exception as error:  # Capture any failure during export
            logger.exception("Failed to export execution history for crew %s", self.name)  # Log the exception details
            return False  # Indicate failure


class CrewFactory:  # Provide convenience constructors for common crew compositions
    """Factory class for creating vetted crew configurations."""  # Document the purpose of the factory

    @staticmethod
    def create_research_crew(  # Build a research-focused crew
        llm: Ollama,  # Supply the Ollama model shared by agents
        topic: str,  # Supply the research topic
        verbose: bool = False,  # Toggle verbose execution logging
    ) -> CustomCrew:  # Return a configured CustomCrew instance
        research_agent = AgentFactory.create_research_agent(llm, verbose)  # Create the research agent persona
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Create the writing agent persona
        research_task = TaskFactory.create_research_task(research_agent.get_agent(), topic)  # Create the research task bound to the research agent
        writing_task = TaskFactory.create_writing_task(writer_agent.get_agent(), topic, "research report")  # Create the writing task bound to the writer agent
        return CustomCrew(  # Assemble the crew using the configured agents and tasks
            name=f"research_crew_{topic.replace(' ', '_').lower()}",  # Generate a deterministic crew name
            agents=[research_agent, writer_agent],  # Provide the agent wrappers
            tasks=[research_task, writing_task],  # Provide the task wrappers
            verbose=verbose,  # Propagate verbosity
            process="sequential",  # Use sequential execution for clarity
        )  # Return the configured research crew

    @staticmethod
    def create_analysis_crew(  # Build an analysis-focused crew
        llm: Ollama,  # Supply the Ollama model shared by agents
        data: str,  # Supply the data description to analyze
        verbose: bool = False,  # Toggle verbose execution logging
    ) -> CustomCrew:  # Return a configured CustomCrew instance
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Create the analyst agent persona
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Create the writing agent persona
        analysis_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), data)  # Create the analysis task bound to the analyst agent
        summary_task = TaskFactory.create_summary_task(writer_agent.get_agent(), "analysis results")  # Create the summary task bound to the writer agent
        return CustomCrew(  # Assemble the crew using the configured agents and tasks
            name=f"analysis_crew_{abs(hash(data))}",  # Generate a deterministic yet obfuscated crew name
            agents=[analyst_agent, writer_agent],  # Provide the agent wrappers
            tasks=[analysis_task, summary_task],  # Provide the task wrappers
            verbose=verbose,  # Propagate verbosity
            process="sequential",  # Use sequential execution for clarity
        )  # Return the configured analysis crew

    @staticmethod
    def create_coding_crew(  # Build a coding-focused crew
        llm: Ollama,  # Supply the Ollama model shared by agents
        requirements: str,  # Supply the coding requirements
        language: str = "python",  # Supply the implementation language preference
        verbose: bool = False,  # Toggle verbose execution logging
    ) -> CustomCrew:  # Return a configured CustomCrew instance
        coder_agent = AgentFactory.create_coder_agent(llm, verbose)  # Create the coding agent persona
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Create the analyst agent persona for review
        coding_task = TaskFactory.create_coding_task(coder_agent.get_agent(), requirements, language)  # Create the coding task bound to the coder agent
        review_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), "code review")  # Create the review task bound to the analyst agent
        return CustomCrew(  # Assemble the crew using the configured agents and tasks
            name=f"coding_crew_{requirements.replace(' ', '_').lower()[:20]}",  # Generate a deterministic crew name limited in length
            agents=[coder_agent, analyst_agent],  # Provide the agent wrappers
            tasks=[coding_task, review_task],  # Provide the task wrappers
            verbose=verbose,  # Propagate verbosity
            process="sequential",  # Use sequential execution for clarity
        )  # Return the configured coding crew
