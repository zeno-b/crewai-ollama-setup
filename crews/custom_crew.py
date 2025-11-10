from typing import Any, Dict, List, Optional  # Provide typing helpers for clarity and static analysis
import asyncio  # Provide asynchronous utilities for non-blocking execution
import json  # Provide JSON serialization for exporting execution history
import logging  # Provide logging utilities for observability within crew abstractions
from datetime import datetime, timezone  # Provide timezone-aware timestamps for audit trails

from crewai import Agent, Crew, Task  # Provide CrewAI primitives leveraged by this wrapper
from langchain_community.llms import Ollama  # Provide Ollama type hints for manager integrations

from agents.custom_agent import AgentFactory, CustomAgent  # Import agent wrappers with explicit comments
from tasks.custom_task import CustomTask, TaskFactory  # Import task wrappers with explicit comments

logger = logging.getLogger(__name__)  # Acquire module-specific logger for contextualized diagnostics


class CustomCrew:  # Provide secure, validated wrapper around CrewAI Crew
    def __init__(  # Initialize wrapper with explicit configuration parameters
        self,  # Reference current instance
        name: str,  # Specify unique crew name for tracking
        agents: List[CustomAgent],  # Provide list of configured custom agents
        tasks: List[CustomTask],  # Provide list of configured custom tasks
        verbose: bool = False,  # Control verbosity for debugging or auditing
        process: str = "sequential",  # Specify crew execution mode
        manager_llm: Optional[Ollama] = None,  # Provide optional manager LLM for coordination
        function_calling_llm: Optional[Ollama] = None,  # Provide optional function calling LLM
        config: Optional[Dict[str, Any]] = None,  # Provide optional configuration overrides
        cache: bool = True,  # Control caching behavior for CrewAI
        max_rpm: Optional[int] = None,  # Specify optional rate limit per minute
        share_crew: bool = False,  # Control crew sharing behavior
        output_log_file: Optional[str] = None,  # Provide optional log file target
        embedder_config: Optional[Dict[str, Any]] = None,  # Provide optional embedder configuration
    ) -> None:  # Indicate initializer returns nothing
        self.name = name  # Persist crew name for reuse
        self.agents = agents  # Persist agent list for reuse
        self.tasks = tasks  # Persist task list for reuse
        self.verbose = verbose  # Persist verbosity preference
        self.process = process  # Persist process strategy
        self.manager_llm = manager_llm  # Persist manager LLM reference
        self.function_calling_llm = function_calling_llm  # Persist function calling LLM reference
        self.config = config or {}  # Normalize configuration dictionary
        self.cache = cache  # Persist cache preference
        self.max_rpm = max_rpm  # Persist rate limit configuration
        self.share_crew = share_crew  # Persist crew sharing flag
        self.output_log_file = output_log_file  # Persist log file configuration
        self.embedder_config = embedder_config  # Persist embedder configuration
        self.crew = self._create_crew()  # Instantiate underlying CrewAI crew during initialization
        self.execution_history: List[Dict[str, Any]] = []  # Initialize execution history for auditing
        self.current_execution: Optional[Dict[str, Any]] = None  # Initialize placeholder for current execution context

    def _create_crew(self) -> Crew:  # Provide helper to instantiate wrapped CrewAI crew
        return Crew(  # Return new CrewAI crew instance using sanitized configuration
            agents=[agent.get_agent() for agent in self.agents],  # Provide underlying CrewAI agents
            tasks=[task.get_task() for task in self.tasks],  # Provide underlying CrewAI tasks
            verbose=self.verbose,  # Supply verbosity preference
            process=self.process,  # Supply process strategy
            manager_llm=self.manager_llm,  # Supply manager LLM reference
            function_calling_llm=self.function_calling_llm,  # Supply function calling LLM reference
            config=self.config,  # Supply configuration overrides
            cache=self.cache,  # Supply cache preference
            max_rpm=self.max_rpm,  # Supply rate limit configuration
            share_crew=self.share_crew,  # Supply crew sharing flag
            output_log_file=self.output_log_file,  # Supply log file configuration
            embedder_config=self.embedder_config,  # Supply embedder configuration
        )

    def get_crew_info(self) -> Dict[str, Any]:  # Provide serializable snapshot of crew configuration
        return {  # Return dictionary representing crew configuration
            "name": self.name,  # Include crew name
            "agents": [agent.get_agent_info() for agent in self.agents],  # Include detailed agent information
            "tasks": [task.get_task_info() for task in self.tasks],  # Include detailed task information
            "process": self.process,  # Include process strategy
            "verbose": self.verbose,  # Include verbosity preference
            "cache": self.cache,  # Include cache configuration
            "max_rpm": self.max_rpm,  # Include rate limit configuration
            "execution_history": self.execution_history,  # Include execution history for transparency
        }

    def add_agent(self, agent: CustomAgent) -> None:  # Provide method to add agent securely
        self.agents.append(agent)  # Append new agent to crew
        self.crew = self._create_crew()  # Recreate underlying CrewAI crew to reflect change
        logger.info("Agent %s added to crew %s", agent.name, self.name)  # Emit informational log capturing update

    def remove_agent(self, agent_name: str) -> bool:  # Provide method to remove agent securely
        original_count = len(self.agents)  # Capture original agent count for comparison
        self.agents = [agent for agent in self.agents if agent.name != agent_name]  # Filter out specified agent
        if len(self.agents) < original_count:  # Detect whether removal occurred
            self.crew = self._create_crew()  # Recreate underlying CrewAI crew to reflect change
            logger.info("Agent %s removed from crew %s", agent_name, self.name)  # Emit informational log capturing update
            return True  # Indicate removal success
        logger.warning("Agent %s not found in crew %s", agent_name, self.name)  # Emit warning when agent absent
        return False  # Indicate removal failure

    def add_task(self, task: CustomTask) -> None:  # Provide method to add task securely
        self.tasks.append(task)  # Append new task to crew
        self.crew = self._create_crew()  # Recreate underlying CrewAI crew to reflect change
        logger.info("Task %s added to crew %s", task.name, self.name)  # Emit informational log capturing update

    def remove_task(self, task_name: str) -> bool:  # Provide method to remove task securely
        original_count = len(self.tasks)  # Capture original task count for comparison
        self.tasks = [task for task in self.tasks if task.name != task_name]  # Filter out specified task
        if len(self.tasks) < original_count:  # Detect whether removal occurred
            self.crew = self._create_crew()  # Recreate underlying CrewAI crew to reflect change
            logger.info("Task %s removed from crew %s", task_name, self.name)  # Emit informational log capturing update
            return True  # Indicate removal success
        logger.warning("Task %s not found in crew %s", task_name, self.name)  # Emit warning when task absent
        return False  # Indicate removal failure

    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Provide synchronous execution helper
        execution_id = f"{self.name}_{datetime.now(timezone.utc).isoformat()}"  # Generate unique execution identifier
        try:  # Provide error handling for execution
            logger.info("Crew %s starting execution %s", self.name, execution_id)  # Emit informational log before execution
            start_time = datetime.now(timezone.utc)  # Capture execution start timestamp
            result = self.crew.kickoff(inputs=inputs)  # Execute crew synchronously
            end_time = datetime.now(timezone.utc)  # Capture execution end timestamp
            duration = (end_time - start_time).total_seconds()  # Calculate execution duration in seconds
            record = {  # Prepare execution record for auditing
                "execution_id": execution_id,  # Persist execution identifier
                "start_time": start_time.isoformat(),  # Persist start timestamp
                "end_time": end_time.isoformat(),  # Persist end timestamp
                "duration": duration,  # Persist duration
                "inputs": inputs,  # Persist provided inputs
                "result": str(result),  # Persist execution result
                "status": "success",  # Persist status indicator
            }
            self.execution_history.append(record)  # Append record to execution history
            self.current_execution = record  # Persist current execution context
            return record  # Return execution record to caller
        except Exception as exc:  # Catch unexpected failures during execution
            logger.exception("Crew %s execution %s failed: %s", self.name, execution_id, exc)  # Emit exception log capturing context
            failure_record = {  # Prepare failure record for auditing
                "execution_id": execution_id,  # Persist execution identifier
                "start_time": datetime.now(timezone.utc).isoformat(),  # Persist failure timestamp
                "end_time": datetime.now(timezone.utc).isoformat(),  # Persist failure timestamp again
                "duration": 0,  # Persist zero duration due to failure
                "inputs": inputs,  # Persist provided inputs
                "error": str(exc),  # Persist failure message
                "status": "failed",  # Persist status indicator
            }
            self.execution_history.append(failure_record)  # Append failure record to history
            self.current_execution = failure_record  # Persist failure context
            return failure_record  # Return failure information to caller

    async def execute_async(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # Provide asynchronous execution helper
        loop = asyncio.get_event_loop()  # Acquire current event loop reference
        return await loop.run_in_executor(None, self.execute, inputs)  # Delegate synchronous execution to thread pool without blocking event loop

    def validate_crew(self) -> bool:  # Provide validation routine for configuration completeness
        if not self.agents:  # Ensure crew contains agents
            logger.error("Crew %s validation failed: no agents configured", self.name)  # Emit error log for missing agents
            return False  # Indicate validation failure
        if not self.tasks:  # Ensure crew contains tasks
            logger.error("Crew %s validation failed: no tasks configured", self.name)  # Emit error log for missing tasks
            return False  # Indicate validation failure
        for agent in self.agents:  # Iterate through configured agents
            if not agent.validate_agent():  # Validate agent configuration
                logger.error("Crew %s validation failed due to invalid agent %s", self.name, agent.name)  # Emit error log for invalid agent
                return False  # Indicate validation failure
        for task in self.tasks:  # Iterate through configured tasks
            if not task.validate_task():  # Validate task configuration
                logger.error("Crew %s validation failed due to invalid task %s", self.name, task.name)  # Emit error log for invalid task
                return False  # Indicate validation failure
        return True  # Indicate validation success

    def get_execution_history(self) -> List[Dict[str, Any]]:  # Provide accessor for execution history
        return self.execution_history  # Return stored execution history data

    def export_execution_history(self, file_path: str) -> bool:  # Provide helper to export execution history securely
        try:  # Provide error handling for export operation
            with open(file_path, "w", encoding="utf-8") as handle:  # Open destination file with explicit encoding
                json.dump(self.execution_history, handle, indent=2)  # Serialize execution history to JSON with indentation
            logger.info("Crew %s execution history exported to %s", self.name, file_path)  # Emit informational log confirming export
            return True  # Indicate export success
        except Exception as exc:  # Catch unexpected failures during export
            logger.exception("Crew %s failed to export execution history: %s", self.name, exc)  # Emit exception log capturing context
            return False  # Indicate export failure


class CrewFactory:  # Provide factory methods to create standardized crew templates
    @staticmethod  # Declare method does not use instance or class state
    def create_research_crew(llm: Ollama, topic: str, verbose: bool = False) -> CustomCrew:  # Provide helper for research crew template
        research_agent = AgentFactory.create_research_agent(llm, verbose)  # Instantiate research-focused agent
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Instantiate writer-focused agent
        research_task = TaskFactory.create_research_task(research_agent.get_agent(), topic)  # Instantiate research task tied to research agent
        writing_task = TaskFactory.create_writing_task(writer_agent.get_agent(), topic, "research report")  # Instantiate writing task tied to writer agent
        crew_name = f"research_crew_{topic.replace(' ', '_').lower()}"[:64]  # Generate deterministic crew name with safe length
        return CustomCrew(  # Instantiate CustomCrew with research-focused configuration
            name=crew_name,  # Provide generated crew name
            agents=[research_agent, writer_agent],  # Provide participating agents
            tasks=[research_task, writing_task],  # Provide associated tasks
            verbose=verbose,  # Propagate verbosity preference
            process="sequential",  # Use sequential process to maintain order
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_analysis_crew(llm: Ollama, data: str, verbose: bool = False) -> CustomCrew:  # Provide helper for analysis crew template
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Instantiate analyst-focused agent
        writer_agent = AgentFactory.create_writer_agent(llm, verbose)  # Instantiate writer-focused agent
        analysis_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), data)  # Instantiate analysis task tied to analyst agent
        summary_task = TaskFactory.create_summary_task(writer_agent.get_agent(), "analysis results")  # Instantiate summary task tied to writer agent
        crew_name = f"analysis_crew_{abs(hash(data))}"[:64]  # Generate deterministic crew name with safe length
        return CustomCrew(  # Instantiate CustomCrew with analysis-focused configuration
            name=crew_name,  # Provide generated crew name
            agents=[analyst_agent, writer_agent],  # Provide participating agents
            tasks=[analysis_task, summary_task],  # Provide associated tasks
            verbose=verbose,  # Propagate verbosity preference
            process="sequential",  # Use sequential process to maintain order
        )

    @staticmethod  # Declare method does not use instance or class state
    def create_coding_crew(  # Provide helper for coding crew template
        llm: Ollama,  # Provide shared Ollama client
        requirements: str,  # Provide textual requirements for coding task
        language: str = "python",  # Provide programming language hint
        verbose: bool = False,  # Provide verbosity preference
    ) -> CustomCrew:  # Indicate factory returns CustomCrew instance
        coder_agent = AgentFactory.create_coder_agent(llm, verbose)  # Instantiate coder-focused agent
        analyst_agent = AgentFactory.create_analyst_agent(llm, verbose)  # Instantiate analyst-focused agent for review
        coding_task = TaskFactory.create_coding_task(coder_agent.get_agent(), requirements, language)  # Instantiate coding task tied to coder agent
        review_task = TaskFactory.create_analysis_task(analyst_agent.get_agent(), "code review")  # Instantiate review task tied to analyst agent
        crew_name = f"coding_crew_{abs(hash(requirements))}"[:64]  # Generate deterministic crew name with safe length
        return CustomCrew(  # Instantiate CustomCrew with coding-focused configuration
            name=crew_name,  # Provide generated crew name
            agents=[coder_agent, analyst_agent],  # Provide participating agents
            tasks=[coding_task, review_task],  # Provide associated tasks
            verbose=verbose,  # Propagate verbosity preference
            process="sequential",  # Use sequential process to maintain order
        )
