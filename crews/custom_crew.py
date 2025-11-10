import asyncio  # Import asyncio to support asynchronous execution pathways
import json  # Import json to serialize execution history when exporting
import logging  # Import logging to emit structured diagnostic messages
from datetime import datetime  # Import datetime to timestamp executions
from typing import Any, Dict, List, Optional  # Import typing helpers to annotate crew metadata

from crewai import Crew, Task  # Import CrewAI Crew and Task classes to orchestrate execution
from langchain_community.llms import Ollama  # Import Ollama type to annotate language model dependencies

from agents.custom_agent import AgentFactory, CustomAgent  # Import agent factory and wrapper utilities using absolute imports
from config.settings import settings  # Import project settings to align behavior with configuration
from tasks.custom_task import CustomTask, TaskFactory  # Import task wrappers and factories for reuse

logger = logging.getLogger(__name__)  # Configure a module-level logger for crew lifecycle events


class CustomCrew:  # Define a wrapper around CrewAI's Crew with enhanced metadata management
    def __init__(  # Initialize the crew wrapper and instantiate the underlying CrewAI crew
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
    ) -> None:
        self.name = name  # Store the crew name for identification
        self.agents = agents  # Store the list of participating custom agents
        self.tasks = tasks  # Store the list of custom tasks assigned to the crew
        self.verbose = verbose  # Store whether verbose execution logs are desired
        self.process = process  # Store the execution strategy (sequential or parallel)
        self.manager_llm = manager_llm  # Store an optional manager LLM
        self.function_calling_llm = function_calling_llm  # Store an optional function-calling LLM
        self.config = config or {}  # Store additional configuration parameters with a safe default
        self.cache = cache  # Store whether CrewAI caching should be enabled
        self.max_rpm = max_rpm  # Store the optional per-minute rate limit for the crew
        self.share_crew = share_crew  # Store whether the crew state is shareable across runs
        self.output_log_file = output_log_file  # Store the optional log file path for outputs
        self.embedder_config = embedder_config  # Store optional embedder configuration details
        self.crew = self._create_crew()  # Instantiate the underlying CrewAI crew using the stored configuration
        self.execution_history: List[Dict[str, Any]] = []  # Initialize in-memory execution history tracking

    def _create_crew(self) -> Crew:  # Internal helper to instantiate a CrewAI crew from current state
        return Crew(  # Create the CrewAI crew instance
            agents=[agent.get_agent() for agent in self.agents],  # Supply the underlying CrewAI agents
            tasks=[task.get_task() for task in self.tasks],  # Supply the underlying CrewAI tasks
            verbose=self.verbose,  # Apply the verbose flag
            process=self.process,  # Apply the execution strategy
            manager_llm=self.manager_llm,  # Supply the optional manager LLM
            function_calling_llm=self.function_calling_llm,  # Supply the optional function-calling LLM
            config=self.config,  # Supply additional configuration parameters
            cache=self.cache,  # Apply caching preference
            max_rpm=self.max_rpm,  # Apply the per-minute rate limit
            share_crew=self.share_crew,  # Apply the crew sharing preference
            output_log_file=self.output_log_file,  # Apply the optional output log file path
            embedder_config=self.embedder_config  # Apply the optional embedder configuration
        )

    def get_crew_info(self) -> Dict[str, Any]:  # Produce a serializable snapshot of the crew configuration
        return {  # Return a dictionary describing the crew state
            "name": self.name,  # Include the crew name
            "agents": [agent.get_agent_info() for agent in self.agents],  # Include metadata for each agent
            "tasks": [task.get_task_info() for task in self.tasks],  # Include metadata for each task
            "process": self.process,  # Include the execution strategy
            "verbose": self.verbose,  # Include the verbose flag
            "cache": self.cache,  # Include the caching preference
            "max_rpm": self.max_rpm,  # Include the per-minute rate limit
            "execution_history": self.execution_history  # Include historical execution records
        }

    def add_agent(self, agent: CustomAgent) -> None:  # Add a custom agent to the crew and rebuild the CrewAI instance
        self.agents.append(agent)  # Append the new agent to the crew
        self.crew = self._create_crew()  # Rebuild the CrewAI crew to include the new agent
        logger.info("Added agent %s to crew %s", agent.name, self.name)  # Log the addition for auditing

    def remove_agent(self, agent_name: str) -> bool:  # Remove a custom agent by name and rebuild the CrewAI instance
        initial_count = len(self.agents)  # Capture the agent count before removal
        self.agents = [agent for agent in self.agents if agent.name != agent_name]  # Filter out the target agent
        if len(self.agents) < initial_count:  # Determine whether an agent was removed
            self.crew = self._create_crew()  # Rebuild the CrewAI crew without the removed agent
            logger.info("Removed agent %s from crew %s", agent_name, self.name)  # Log the removal event
            return True  # Indicate that an agent was removed
        logger.warning("Agent %s not found in crew %s", agent_name, self.name)  # Warn when the agent was not found
        return False  # Indicate that no agent was removed

    def add_task(self, task: CustomTask) -> None:  # Add a custom task to the crew and rebuild the CrewAI instance
        self.tasks.append(task)  # Append the new task to the crew
        self.crew = self._create_crew()  # Rebuild the CrewAI crew to include the new task
        logger.info("Added task %s to crew %s", task.name, self.name)  # Log the addition for auditing

    def remove_task(self, task_name: str) -> bool:  # Remove a custom task by name and rebuild the CrewAI instance
        initial_count = len(self.tasks)  # Capture the task count before removal
        self.tasks = [task for task in self.tasks if task.name != task_name]  # Filter out the target task
        if len(self.tasks) < initial_count:  # Determine whether a task was removed
            self.crew = self._create_crew()  # Rebuild the CrewAI crew without the removed task
            logger.info("Removed task %s from crew %s", task_name, self.name)  # Log the removal event
            return True  # Indicate that a task was removed
        logger.warning("Task %s not found in crew %s", task_name, self.name)  # Warn when the task was not found
        return False  # Indicate that no task was removed

    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Execute the crew synchronously and record history
        execution_id = f"{self.name}_{datetime.utcnow().isoformat()}"  # Generate a unique execution identifier
        try:  # Attempt to execute the crew workflow
            logger.info("Starting crew execution %s", execution_id)  # Log the start of execution
            execution_start = datetime.utcnow()  # Record the start time
            result = self.crew.kickoff(inputs=inputs)  # Execute the crew synchronously
            execution_end = datetime.utcnow()  # Record the end time
            execution_record = {  # Construct a record describing the execution
                "execution_id": execution_id,  # Store the unique execution identifier
                "start_time": execution_start.isoformat(),  # Store the start time
                "end_time": execution_end.isoformat(),  # Store the end time
                "duration": (execution_end - execution_start).total_seconds(),  # Store the execution duration in seconds
                "inputs": inputs,  # Store the inputs passed to the crew
                "result": str(result),  # Store a textual representation of the result
                "status": "success"  # Mark the execution as successful
            }  # Close the execution record dictionary
            self.execution_history.append(execution_record)  # Append the record to the execution history
            logger.info("Crew execution %s completed successfully", execution_id)  # Log successful completion
            return {  # Return a structured summary to the caller
                "execution_id": execution_id,  # Include the execution identifier
                "result": result,  # Include the raw result
                "duration": execution_record["duration"],  # Include the execution duration
                "status": "success"  # Include the execution status
            }  # Close the response dictionary
        except Exception as execution_error:  # Handle failures during execution
            logger.error("Crew execution %s failed: %s", execution_id, execution_error)  # Log the failure details
            failure_record = {  # Construct a record describing the failed execution
                "execution_id": execution_id,  # Store the unique execution identifier
                "start_time": datetime.utcnow().isoformat(),  # Store the failure timestamp
                "end_time": datetime.utcnow().isoformat(),  # Store the completion timestamp
                "duration": 0,  # Set the duration to zero for failed executions
                "inputs": inputs,  # Store the inputs passed to the crew
                "error": str(execution_error),  # Store the error description
                "status": "failed"  # Mark the execution as failed
            }  # Close the failure record dictionary
            self.execution_history.append(failure_record)  # Append the failure record to the execution history
            return {  # Return a structured failure summary to the caller
                "execution_id": execution_id,  # Include the execution identifier
                "error": str(execution_error),  # Include the error description
                "status": "failed"  # Include the execution status
            }  # Close the failure response dictionary

    async def execute_async(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Execute the crew asynchronously without blocking the event loop
        return await asyncio.to_thread(self.execute, inputs)  # Run the synchronous execute method in a worker thread

    def validate_crew(self) -> bool:  # Validate that the crew has sufficient configuration to execute
        if not self.agents:  # Ensure at least one agent is configured
            logger.error("Crew %s has no configured agents", self.name)  # Log the configuration issue
            return False  # Indicate validation failure
        if not self.tasks:  # Ensure at least one task is configured
            logger.error("Crew %s has no configured tasks", self.name)  # Log the configuration issue
            return False  # Indicate validation failure
        if len(self.agents) > settings.max_agents:  # Enforce the configured maximum number of agents
            logger.error("Crew %s exceeds max agents limit %d", self.name, settings.max_agents)  # Log the limit violation
            return False  # Indicate validation failure
        if len(self.tasks) > settings.max_tasks:  # Enforce the configured maximum number of tasks
            logger.error("Crew %s exceeds max tasks limit %d", self.name, settings.max_tasks)  # Log the limit violation
            return False  # Indicate validation failure
        for agent in self.agents:  # Validate each agent configuration
            if not agent.validate_agent():  # Check whether the agent is valid
                logger.error("Crew %s contains invalid agent %s", self.name, agent.name)  # Log the invalid agent
                return False  # Indicate validation failure
        for task in self.tasks:  # Validate each task configuration
            if not task.validate_task():  # Check whether the task is valid
                logger.error("Crew %s contains invalid task %s", self.name, task.name)  # Log the invalid task
                return False  # Indicate validation failure
        return True  # Indicate that the crew configuration is valid

    def get_execution_history(self) -> List[Dict[str, Any]]:  # Provide access to the recorded execution history
        return self.execution_history  # Return the in-memory execution history

    def export_execution_history(self, file_path: str) -> bool:  # Export the execution history to a JSON file
        try:  # Attempt to write the execution history to disk
            with open(file_path, "w", encoding="utf-8") as export_file:  # Open the target file using UTF-8 encoding
                json.dump(self.execution_history, export_file, indent=2)  # Serialize the execution history with indentation
            logger.info("Exported crew %s execution history to %s", self.name, file_path)  # Log the successful export
            return True  # Indicate success
        except Exception as export_error:  # Handle failures when writing to disk
            logger.error("Failed to export execution history for crew %s: %s", self.name, export_error)  # Log the failure details
            return False  # Indicate failure


class CrewFactory:  # Provide factory helpers to construct common crew archetypes
    @staticmethod  # Indicate that the method does not rely on instance state
    def create_research_crew(llm: Ollama, topic: str, verbose: bool = False) -> CustomCrew:  # Build a crew specialized in research
        research_agent = AgentFactory.create_research_agent(llm, verbose)  # Instantiate a research-focused agent
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Instantiate a writing-focused agent
        research_task = TaskFactory.create_research_task(research_agent.get_agent(), topic)  # Create a research task for the topic
        writing_task = TaskFactory.create_writing_task(writer_agent.get_agent(), topic, "research report")  # Create a writing task for the findings
        return CustomCrew(  # Assemble the crew using the prepared agents and tasks
            name=f"research_crew_{topic.replace(' ', '_').lower()}",  # Generate a descriptive crew name
            agents=[research_agent, writer_agent],  # Supply the participating agents
            tasks=[research_task, writing_task],  # Supply the tasks to execute
            verbose=verbose,  # Apply the verbosity preference
            process="sequential"  # Execute tasks sequentially to ensure research precedes writing
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_analysis_crew(llm: Ollama, data: str, verbose: bool = False) -> CustomCrew:  # Build a crew specialized in analysis
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Instantiate an analytical agent
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Instantiate a writing-focused agent
        analysis_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), data)  # Create an analysis task for the provided data
        summary_task = TaskFactory.create_summary_task(writer_agent.get_agent(), "analysis results")  # Create a summary task to communicate findings
        return CustomCrew(  # Assemble the crew using the prepared agents and tasks
            name=f"analysis_crew_{abs(hash(data))}",  # Generate a descriptive crew name based on the data hash
            agents=[analyst_agent, writer_agent],  # Supply the participating agents
            tasks=[analysis_task, summary_task],  # Supply the tasks to execute
            verbose=verbose,  # Apply the verbosity preference
            process="sequential"  # Execute tasks sequentially so analysis precedes summarization
        )

    @staticmethod  # Indicate that the method does not rely on instance state
    def create_coding_crew(llm: Ollama, requirements: str, language: str = "python", verbose: bool = False) -> CustomCrew:  # Build a crew specialized in coding
        coder_agent = AgentFactory.create_coder_agent(llm, verbose)  # Instantiate a coding-focused agent
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Instantiate an analytical agent for review
        coding_task = TaskFactory.create_coding_task(coder_agent.get_agent(), requirements, language)  # Create a coding task for the requirements
        review_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), "code review")  # Create a review task to assess the produced code
        return CustomCrew(  # Assemble the crew using the prepared agents and tasks
            name=f"coding_crew_{abs(hash(requirements))}",  # Generate a descriptive crew name based on the requirements hash
            agents=[coder_agent, analyst_agent],  # Supply the participating agents
            tasks=[coding_task, review_task],  # Supply the tasks to execute
            verbose=verbose,  # Apply the verbosity preference
            process="sequential"  # Execute tasks sequentially to enforce implementation followed by review
        )
