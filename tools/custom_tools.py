from __future__ import annotations  # Enable postponed evaluation of annotations for forward references

import json  # Import json to serialize and deserialize structured data safely
import logging  # Import logging to provide security-auditable diagnostics
import os  # Import os for environment and filesystem interactions
import re  # Import re to validate inputs with regular expressions
import subprocess  # Import subprocess to execute external commands in a controlled way
import sys  # Import sys to access the current Python executable for sandboxed runs
import tempfile  # Import tempfile to create secure temporary files
from pathlib import Path  # Import Path to handle filesystem paths safely
from typing import Any, Dict, List, Optional, Type  # Import typing helpers for clarity and static analysis
from urllib.parse import urlparse  # Import urlparse to validate URLs rigorously

import requests  # Import requests to issue outbound HTTP calls with proper timeouts
from crewai_tools import BaseTool  # Import BaseTool as the base class for CrewAI tools
from pydantic import BaseModel, Field, field_validator  # Import Pydantic utilities for input validation

logger = logging.getLogger(__name__)  # Initialize a module-level logger for structured logging

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path.cwd())).resolve()  # Resolve the workspace root to restrict filesystem access
SAFE_PATH_PATTERN = re.compile(r"^[\w\-./ ]+$")  # Define a conservative regex to validate relative file paths
ALLOWED_HTTP_SCHEMES = {"http", "https"}  # Restrict HTTP requests to safe schemes
MAX_WEB_RESULTS = 10  # Define an upper bound for returned web search results to avoid overload


def resolve_secure_path(raw_path: str) -> Path:  # Resolve and validate filesystem paths against the workspace boundary
    candidate_path = Path(raw_path).expanduser().resolve()  # Resolve the candidate path to an absolute location
    if not SAFE_PATH_PATTERN.match(raw_path):  # Reject path strings containing suspicious characters
        raise ValueError("Path contains unsupported characters.")  # Enforce strong input validation
    if not str(candidate_path).startswith(str(WORKSPACE_ROOT)):  # Ensure the target path resides within the workspace
        raise ValueError("Access to the requested path is not permitted.")  # Prevent directory traversal attacks
    return candidate_path  # Return the validated path for use


class WebSearchInput(BaseModel):  # Define the validated input schema for the web search tool
    query: str = Field(..., description="Search query")  # Capture the search query string
    max_results: int = Field(default=5, description="Maximum number of results to return")  # Limit the number of results returned
    search_engine: str = Field(default="duckduckgo", description="Search engine identifier")  # Capture the selected search engine identifier

    @field_validator("max_results")  # Validate the requested maximum result count
    @classmethod
    def enforce_result_bounds(cls, value: int) -> int:  # Ensure the requested count stays within safe bounds
        if value < 1:  # Reject non-positive result counts
            raise ValueError("max_results must be at least 1")  # Provide a descriptive validation error
        return min(value, MAX_WEB_RESULTS)  # Cap the result count to protect resources


class WebSearchTool(BaseTool):  # Implement a mock web search tool suitable for testing
    name: str = "Web Search"  # Expose the tool's name for CrewAI registration
    description: str = "Search the web for information"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = WebSearchInput  # Associate the validated input schema with the tool

    def _run(self, query: str, max_results: int = 5, search_engine: str = "duckduckgo") -> str:  # Execute the web search logic
        try:  # Wrap execution in a try block to handle unexpected errors
            sanitized_query = query.strip()  # Remove leading and trailing whitespace from the query
            if not sanitized_query:  # Ensure the query is not empty after sanitization
                raise ValueError("Query must not be empty.")  # Enforce meaningful input
            results = [  # Build a deterministic mock response list
                {
                    "title": f"Result {index + 1} for {sanitized_query}",  # Provide a mock result title
                    "url": f"https://example.com/{search_engine}/{index + 1}",  # Provide a mock URL referencing the chosen engine
                    "snippet": f"Sample snippet for {sanitized_query} returned by {search_engine}.",  # Provide a mock snippet summary
                }
                for index in range(min(max_results, MAX_WEB_RESULTS))  # Generate up to the allowed number of results
            ]  # Close the mock result list
            return json.dumps(results, indent=2)  # Serialize the mock results as formatted JSON
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("Web search failed for query %s", query)  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class FileReadInput(BaseModel):  # Define the validated input schema for the file reader tool
    file_path: str = Field(..., description="Path to the file to read")  # Capture the file path requested by the caller
    encoding: str = Field(default="utf-8", description="File encoding")  # Capture the desired text encoding


class FileReadTool(BaseTool):  # Implement a safe file reader tool
    name: str = "File Reader"  # Expose the tool's name for CrewAI registration
    description: str = "Read content from a file"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = FileReadInput  # Associate the validated input schema with the tool

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:  # Execute the file reading logic
        try:  # Wrap execution in a try block to handle unexpected errors
            secure_path = resolve_secure_path(file_path)  # Resolve and validate the requested path
            if not secure_path.exists():  # Check whether the file exists
                return f"Error: File not found: {secure_path}"  # Return a descriptive error message when missing
            if not secure_path.is_file():  # Ensure the path points to a regular file
                return f"Error: Path is not a file: {secure_path}"  # Prevent directory reads
            return secure_path.read_text(encoding=encoding)  # Read and return the file contents using the requested encoding
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("File read failed for %s", file_path)  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class FileWriteInput(BaseModel):  # Define the validated input schema for the file writer tool
    file_path: str = Field(..., description="Path to the file to write")  # Capture the destination file path
    content: str = Field(..., description="Content to write")  # Capture the content to be persisted
    encoding: str = Field(default="utf-8", description="File encoding")  # Capture the desired text encoding


class FileWriteTool(BaseTool):  # Implement a safe file writer tool
    name: str = "File Writer"  # Expose the tool's name for CrewAI registration
    description: str = "Write content to a file"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = FileWriteInput  # Associate the validated input schema with the tool

    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:  # Execute the file writing logic
        try:  # Wrap execution in a try block to handle unexpected errors
            secure_path = resolve_secure_path(file_path)  # Resolve and validate the destination path
            secure_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the destination directory exists
            secure_path.write_text(content, encoding=encoding)  # Write the provided content using the desired encoding
            return f"Successfully wrote to file: {secure_path}"  # Return a success message including the resolved path
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("File write failed for %s", file_path)  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class CodeExecuteInput(BaseModel):  # Define the validated input schema for the code execution tool
    code: str = Field(..., description="Code to execute")  # Capture the code snippet to execute
    language: str = Field(default="python", description="Programming language")  # Capture the chosen programming language
    timeout: int = Field(default=30, description="Execution timeout in seconds")  # Capture the maximum execution duration in seconds

    @field_validator("timeout")  # Validate the requested timeout
    @classmethod
    def enforce_timeout_bounds(cls, value: int) -> int:  # Ensure the timeout is within a safe range
        if value <= 0:  # Reject non-positive timeouts
            raise ValueError("Timeout must be greater than zero.")  # Provide a descriptive validation error
        return min(value, 120)  # Cap the timeout to prevent long-running processes


class CodeExecuteTool(BaseTool):  # Implement a constrained Python code execution tool
    name: str = "Code Executor"  # Expose the tool's name for CrewAI registration
    description: str = "Execute code in a sandboxed Python environment"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = CodeExecuteInput  # Associate the validated input schema with the tool

    def _run(self, code: str, language: str = "python", timeout: int = 30) -> str:  # Execute the code snippet
        if language.lower() != "python":  # Restrict execution to Python only
            return "Error: Only Python execution is supported for security reasons."  # Return a descriptive error for unsupported languages
        try:  # Wrap execution in a try block to ensure cleanup occurs
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:  # Create a temporary file to store the code
                temp_file.write(code)  # Write the supplied code to the temporary file
                temp_file_path = Path(temp_file.name)  # Capture the path to the temporary file
            safe_env = {"PYTHONPATH": "", "PYTHONWARNINGS": "ignore"}  # Provide a minimal environment to reduce attack surface
            try:  # Attempt to execute the code within the sandbox
                completed = subprocess.run(  # Invoke the Python interpreter in isolated mode
                    [sys.executable, "-I", str(temp_file_path)],  # Execute the temporary script using the isolated interpreter flag
                    capture_output=True,  # Capture stdout and stderr for reporting
                    text=True,  # Decode output as text for readability
                    timeout=timeout,  # Enforce the caller-specified timeout
                    cwd=str(WORKSPACE_ROOT),  # Restrict execution to the workspace directory
                    env=safe_env,  # Supply the minimized environment
                    check=False,  # Avoid raising automatically to allow custom error reporting
                )  # Finish invoking the subprocess
            finally:  # Ensure temporary files are removed regardless of execution outcome
                temp_file_path.unlink(missing_ok=True)  # Delete the temporary script file if it still exists
            if completed.returncode == 0:  # Check whether the subprocess succeeded
                return completed.stdout or "Execution completed with no output."  # Return captured stdout or a default success message
            return f"Error: {completed.stderr.strip() or 'Unknown execution failure.'}"  # Return stderr content as an error message
        except subprocess.TimeoutExpired:  # Handle subprocess timeout separately for clarity
            return f"Error: Code execution timed out after {timeout} seconds."  # Return a timeout-specific error message
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("Code execution failed.")  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class APIRequestInput(BaseModel):  # Define the validated input schema for the API request tool
    url: str = Field(..., description="API endpoint URL")  # Capture the target URL
    method: str = Field(default="GET", description="HTTP method")  # Capture the HTTP method verb
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")  # Capture optional HTTP headers
    data: Optional[Dict[str, Any]] = Field(default=None, description="JSON payload for mutating requests")  # Capture optional JSON payloads
    timeout: int = Field(default=30, description="Request timeout in seconds")  # Capture the maximum request duration in seconds

    @field_validator("method")  # Normalize and validate the HTTP method
    @classmethod
    def normalize_method(cls, value: str) -> str:  # Ensure the HTTP method is uppercase and recognized
        normalized = value.upper()  # Convert to uppercase for consistent comparisons
        if normalized not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:  # Restrict to a safe subset of methods
            raise ValueError("Unsupported HTTP method requested.")  # Reject unsafe methods
        return normalized  # Return the normalized method


class APIRequestTool(BaseTool):  # Implement a guarded HTTP request tool
    name: str = "API Request"  # Expose the tool's name for CrewAI registration
    description: str = "Make HTTP requests to APIs"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = APIRequestInput  # Associate the validated input schema with the tool

    def _run(  # Execute the outbound HTTP call
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> str:
        try:  # Wrap execution in a try block to capture unexpected issues
            parsed = urlparse(url)  # Parse the provided URL for validation
            if parsed.scheme.lower() not in ALLOWED_HTTP_SCHEMES:  # Ensure the scheme is explicitly allowed
                raise ValueError("Only HTTP and HTTPS schemes are permitted.")  # Reject unsupported schemes
            if not parsed.netloc:  # Ensure the URL includes a network location
                raise ValueError("URL must include a valid host.")  # Reject incomplete URLs
            session = requests.Session()  # Create a session to benefit from connection pooling
            request_kwargs: Dict[str, Any] = {  # Build the request parameters
                "method": method,  # Provide the normalized HTTP method
                "url": url,  # Provide the validated URL
                "timeout": timeout,  # Provide the caller-specified timeout
            }  # Close the request kwargs dictionary
            if headers:  # Attach headers when provided
                request_kwargs["headers"] = headers  # Assign the provided headers
            if data and method in {"POST", "PUT", "PATCH"}:  # Attach JSON payloads when applicable
                request_kwargs["json"] = data  # Assign the JSON payload
            response = session.request(**request_kwargs)  # Issue the HTTP request using the session
            result_payload = {  # Build the response payload
                "status_code": response.status_code,  # Include the HTTP status code
                "headers": dict(response.headers),  # Include response headers
                "content": response.text,  # Include response body text
            }  # Close the response payload
            return json.dumps(result_payload, indent=2)  # Serialize the response payload as formatted JSON
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("API request failed for %s %s", method, url)  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class DataAnalysisInput(BaseModel):  # Define the validated input schema for the data analysis tool
    data: str = Field(..., description="Data to analyze as a JSON string")  # Capture the JSON-serialized dataset
    analysis_type: str = Field(default="summary", description="Type of analysis to perform")  # Capture the requested analysis type
    columns: Optional[List[str]] = Field(default=None, description="Optional column subset to analyze")  # Capture optional column filters


class DataAnalysisTool(BaseTool):  # Implement a structured-data analysis tool using pandas
    name: str = "Data Analyzer"  # Expose the tool's name for CrewAI registration
    description: str = "Analyze structured data"  # Describe the tool's purpose
    args_schema: Type[BaseModel] = DataAnalysisInput  # Associate the validated input schema with the tool

    def _run(self, data: str, analysis_type: str = "summary", columns: Optional[List[str]] = None) -> str:  # Execute the analysis routine
        try:  # Wrap execution in a try block to handle errors gracefully
            import pandas as pd  # Import pandas lazily to avoid unnecessary startup cost
            dataset = json.loads(data)  # Deserialize the dataset from JSON
            dataframe = pd.DataFrame(dataset)  # Construct a DataFrame from the dataset
            if columns:  # Apply column filtering when requested
                dataframe = dataframe[columns]  # Select only the requested columns
            if analysis_type == "summary":  # Execute descriptive statistics analysis
                result = {  # Build the summary payload
                    "shape": dataframe.shape,  # Include the dataframe shape
                    "columns": list(dataframe.columns),  # Include the dataframe column names
                    "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},  # Include column data types
                    "summary": dataframe.describe(include="all").to_dict(),  # Include descriptive statistics
                    "missing_values": dataframe.isnull().sum().to_dict(),  # Include missing value counts
                }  # Close the summary payload
            elif analysis_type == "correlation":  # Execute correlation analysis
                result = {"correlation_matrix": dataframe.corr(numeric_only=True).to_dict()}  # Include the correlation matrix
            else:  # Handle unsupported analysis types
                raise ValueError("Unsupported analysis type requested.")  # Reject unknown requests
            return json.dumps(result, indent=2)  # Serialize the analysis result as formatted JSON
        except Exception as error:  # Capture and log unexpected issues
            logger.exception("Data analysis failed.")  # Record the failure for diagnostics
            return f"Error: {error}"  # Return a user-friendly error message


class ToolFactory:  # Provide helpers to assemble collections of tools
    """Factory class for creating tool instances."""  # Document the purpose of the factory

    @staticmethod
    def get_all_tools() -> List[BaseTool]:  # Return all registered tool instances
        return [  # Build and return the full tool list
            WebSearchTool(),  # Include the web search tool
            FileReadTool(),  # Include the file reader tool
            FileWriteTool(),  # Include the file writer tool
            CodeExecuteTool(),  # Include the code execution tool
            APIRequestTool(),  # Include the API request tool
            DataAnalysisTool(),  # Include the data analysis tool
        ]  # Close the tool list

    @staticmethod
    def get_tool_by_name(name: str) -> Optional[BaseTool]:  # Retrieve a tool instance by case-insensitive name
        for tool in ToolFactory.get_all_tools():  # Iterate over all known tools
            if tool.name.lower() == name.lower():  # Compare names case-insensitively
                return tool  # Return the matching tool
        return None  # Return None when no matching tool is found

    @staticmethod
    def get_tools_by_category(category: str) -> List[BaseTool]:  # Retrieve tool instances belonging to a category
        categories: Dict[str, List[BaseTool]] = {  # Define category mappings to tool instances
            "web": [WebSearchTool()],  # Category for web-related tools
            "file": [FileReadTool(), FileWriteTool()],  # Category for filesystem tools
            "code": [CodeExecuteTool()],  # Category for code execution tools
            "api": [APIRequestTool()],  # Category for HTTP tools
            "data": [DataAnalysisTool()],  # Category for data analysis tools
        }  # Close the category mapping
        return categories.get(category.lower(), [])  # Return the requested category or an empty list if not found


class ToolRegistry:  # Maintain a registry of tool instances for reuse
    """Registry for managing custom tools."""  # Document the purpose of the registry

    def __init__(self) -> None:  # Initialize the registry instance
        self.tools: Dict[str, BaseTool] = {}  # Initialize the internal tool dictionary
        self.load_default_tools()  # Load the default tool set immediately

    def load_default_tools(self) -> None:  # Register the default tool set
        for tool in ToolFactory.get_all_tools():  # Iterate over all default tools
            self.register_tool(tool)  # Register each tool with the registry

    def register_tool(self, tool: BaseTool) -> None:  # Register a new tool instance by name
        self.tools[tool.name] = tool  # Store the tool keyed by its name
        logger.info("Registered tool: %s", tool.name)  # Log the registration event

    def unregister_tool(self, tool_name: str) -> bool:  # Remove a tool from the registry
        if tool_name in self.tools:  # Check whether the tool exists
            del self.tools[tool_name]  # Remove the tool from the registry
            logger.info("Unregistered tool: %s", tool_name)  # Log the removal event
            return True  # Indicate success
        logger.warning("Tool not found: %s", tool_name)  # Log the failed removal attempt
        return False  # Indicate failure

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:  # Retrieve a registered tool by name
        return self.tools.get(tool_name)  # Return the tool instance or None

    def list_tools(self) -> List[str]:  # List the names of all registered tools
        return list(self.tools.keys())  # Return a list of tool names

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:  # Retrieve metadata about a registered tool
        tool = self.get_tool(tool_name)  # Fetch the tool from the registry
        if not tool:  # Check whether the tool exists
            logger.warning("Tool info requested for unknown tool: %s", tool_name)  # Log the missing tool access attempt
            return None  # Return None when tool is missing
        return {  # Build and return the tool metadata
            "name": tool.name,  # Include the tool name
            "description": tool.description,  # Include the tool description
            "args_schema": str(tool.args_schema),  # Include the argument schema string representation
        }  # Close the metadata dictionary
