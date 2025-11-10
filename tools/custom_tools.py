import json  # Import json to serialize and deserialize structured data payloads
import logging  # Import logging to emit structured diagnostic information
import os  # Import os to interact with the filesystem in a portable manner
import subprocess  # Import subprocess to safely invoke external processes for code execution
from datetime import datetime  # Import datetime to timestamp temporary artifacts
from typing import Any, Dict, List, Optional, Type  # Import typing helpers to annotate tool interfaces

import requests  # Import requests to perform HTTP requests for API interactions
from crewai_tools import BaseTool  # Import BaseTool to implement custom CrewAI-compatible tools
from pydantic import BaseModel, Field  # Import BaseModel and Field to define validated tool schemas

logger = logging.getLogger(__name__)  # Acquire a module-level logger shared across tools


class WebSearchInput(BaseModel):  # Define the request schema for the web search tool
    query: str = Field(..., description="Search query")  # Require a search query string
    max_results: int = Field(default=5, description="Maximum number of results")  # Limit the number of returned results
    search_engine: str = Field(default="duckduckgo", description="Search engine to use")  # Allow choosing a search engine label


class WebSearchTool(BaseTool):  # Provide a placeholder web search tool implementation
    name: str = "Web Search"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Search the web for information"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = WebSearchInput  # Associate the tool with its validated argument schema

    def _run(self, query: str, max_results: int = 5, search_engine: str = "duckduckgo") -> str:  # Execute the tool with validated arguments
        try:  # Attempt to build a mock search result set
            search_results = [  # Construct placeholder search results
                {
                    "title": f"Result {index + 1} for {query}",  # Provide a synthetic title for the result
                    "url": f"https://example.com/result{index + 1}",  # Provide a synthetic URL for the result
                    "snippet": f"Mock snippet describing result {index + 1} for {query} via {search_engine}."  # Provide a synthetic description for the result
                }
                for index in range(max_results)  # Generate the requested number of mock results
            ]  # Close the list comprehension generating mock results
            return json.dumps(search_results, indent=2)  # Serialize the mock results as formatted JSON
        except Exception as error:  # Handle unexpected failures gracefully
            logger.error("Web search tool failed: %s", error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller


class FileReadInput(BaseModel):  # Define the request schema for the file reader tool
    file_path: str = Field(..., description="Path to the file to read")  # Require the path to the target file
    encoding: str = Field(default="utf-8", description="File encoding")  # Allow specifying the file encoding


class FileReadTool(BaseTool):  # Provide a tool that reads text files from disk
    name: str = "File Reader"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Read content from a file"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = FileReadInput  # Associate the tool with its validated argument schema

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:  # Execute the tool with validated arguments
        try:  # Attempt to read the file from disk
            if not os.path.exists(file_path):  # Check whether the requested file exists
                return f"Error: File not found: {file_path}"  # Inform the caller when the file cannot be found
            with open(file_path, "r", encoding=encoding) as file_handle:  # Open the file safely using the provided encoding
                return file_handle.read()  # Return the entire file contents as a string
        except Exception as error:  # Handle unexpected I/O failures gracefully
            logger.error("File read failed for %s: %s", file_path, error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller


class FileWriteInput(BaseModel):  # Define the request schema for the file writer tool
    file_path: str = Field(..., description="Path to the file to write")  # Require the destination file path
    content: str = Field(..., description="Content to write")  # Require the textual content to persist
    encoding: str = Field(default="utf-8", description="File encoding")  # Allow specifying the file encoding


class FileWriteTool(BaseTool):  # Provide a tool that writes text files to disk
    name: str = "File Writer"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Write content to a file"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = FileWriteInput  # Associate the tool with its validated argument schema

    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:  # Execute the tool with validated arguments
        try:  # Attempt to write the provided content to disk
            absolute_path = os.path.abspath(file_path)  # Resolve the absolute path to prevent traversal ambiguities
            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)  # Ensure the destination directory exists
            with open(absolute_path, "w", encoding=encoding) as file_handle:  # Open the file safely for writing
                file_handle.write(content)  # Write the provided content to disk
            return f"Successfully wrote to file: {absolute_path}"  # Return a success message with the resolved path
        except Exception as error:  # Handle unexpected I/O failures gracefully
            logger.error("File write failed for %s: %s", file_path, error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller


class CodeExecuteInput(BaseModel):  # Define the request schema for the code execution tool
    code: str = Field(..., description="Code to execute")  # Require the code snippet to run
    language: str = Field(default="python", description="Programming language")  # Allow specifying the target language
    timeout: int = Field(default=30, description="Execution timeout in seconds")  # Limit execution time to mitigate runaway code


class CodeExecuteTool(BaseTool):  # Provide a tool that executes Python code snippets
    name: str = "Code Executor"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Execute code in a safe environment"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = CodeExecuteInput  # Associate the tool with its validated argument schema

    def _run(self, code: str, language: str = "python", timeout: int = 30) -> str:  # Execute the tool with validated arguments
        if language.lower() != "python":  # Restrict execution to Python for safety
            return "Error: Only Python execution is supported"  # Inform the caller about unsupported languages
        temp_file = f"/tmp/code_execute_{datetime.now().timestamp()}.py"  # Generate a unique temporary file path
        try:  # Attempt to persist and execute the provided code snippet
            with open(temp_file, "w", encoding="utf-8") as file_handle:  # Write the code to the temporary file
                file_handle.write(code)  # Persist the code snippet to disk
            result = subprocess.run(  # Execute the temporary Python file securely
                ["python", temp_file],  # Invoke the Python interpreter on the temporary file
                capture_output=True,  # Capture stdout and stderr for return to the caller
                text=True,  # Decode captured output as text
                timeout=timeout,  # Enforce the configured timeout to prevent runaway executions
                check=False  # Prevent subprocess from raising on non-zero exit codes
            )  # Close the subprocess.run invocation
            if result.returncode == 0:  # Check whether execution completed successfully
                return result.stdout or "Execution completed with no output"  # Return stdout or a default message
            return f"Error: {result.stderr.strip()}"  # Return stderr content when execution fails
        except subprocess.TimeoutExpired:  # Handle cases where execution exceeds the allowed timeout
            return f"Error: Code execution timed out after {timeout} seconds"  # Inform the caller about the timeout
        except Exception as error:  # Handle unexpected failures gracefully
            logger.error("Code execution failed: %s", error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller
        finally:  # Ensure temporary artifacts are cleaned up
            if os.path.exists(temp_file):  # Check whether the temporary file still exists
                try:  # Attempt to remove the temporary file
                    os.remove(temp_file)  # Delete the temporary file to avoid clutter
                except OSError as cleanup_error:  # Handle failures during cleanup gracefully
                    logger.warning("Failed to remove temporary file %s: %s", temp_file, cleanup_error)  # Log the cleanup issue


class APIRequestInput(BaseModel):  # Define the request schema for the API request tool
    url: str = Field(..., description="API endpoint URL")  # Require the target URL
    method: str = Field(default="GET", description="HTTP method")  # Allow specifying the HTTP method
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")  # Allow custom headers
    data: Optional[Dict[str, Any]] = Field(default=None, description="JSON payload for the request")  # Allow JSON payloads
    timeout: int = Field(default=30, description="Request timeout in seconds")  # Limit request duration for resilience


class APIRequestTool(BaseTool):  # Provide a tool that issues HTTP requests
    name: str = "API Request"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Make HTTP requests to APIs"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = APIRequestInput  # Associate the tool with its validated argument schema

    def _run(  # Execute the tool with validated arguments
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> str:
        try:  # Attempt to issue the HTTP request
            request_kwargs: Dict[str, Any] = {"url": url, "timeout": timeout}  # Seed the request arguments with required parameters
            if headers:  # Attach headers when provided
                request_kwargs["headers"] = headers  # Include the custom headers
            if data and method.upper() in {"POST", "PUT", "PATCH"}:  # Attach JSON payloads for write operations
                request_kwargs["json"] = data  # Include the JSON payload
            response = requests.request(method.upper(), **request_kwargs)  # Issue the HTTP request using the prepared arguments
            response_payload = {  # Construct a structured response payload
                "status_code": response.status_code,  # Include the HTTP status code
                "headers": dict(response.headers),  # Include response headers for inspection
                "content": response.text  # Include the response body as text
            }  # Close the response payload dictionary
            return json.dumps(response_payload, indent=2)  # Serialize the response payload as formatted JSON
        except Exception as error:  # Handle unexpected network failures gracefully
            logger.error("API request to %s failed: %s", url, error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller


class DataAnalysisInput(BaseModel):  # Define the request schema for the data analysis tool
    data: str = Field(..., description="Data to analyze (JSON string)")  # Require a JSON-encoded dataset
    analysis_type: str = Field(default="summary", description="Type of analysis")  # Allow selecting the analysis mode
    columns: Optional[List[str]] = Field(default=None, description="Columns to analyze")  # Allow narrowing the analysis to specific columns


class DataAnalysisTool(BaseTool):  # Provide a tool that performs simple data analyses using pandas
    name: str = "Data Analyzer"  # Declare the human-friendly name exposed to CrewAI
    description: str = "Analyze structured data"  # Describe the tool's purpose for documentation
    args_schema: Type[BaseModel] = DataAnalysisInput  # Associate the tool with its validated argument schema

    def _run(self, data: str, analysis_type: str = "summary", columns: Optional[List[str]] = None) -> str:  # Execute the tool with validated arguments
        try:  # Attempt to perform the requested analysis
            import pandas as pd  # Import pandas lazily to avoid unnecessary dependency loading
            dataset = json.loads(data)  # Deserialize the JSON dataset
            dataframe = pd.DataFrame(dataset)  # Construct a DataFrame from the dataset
            if columns:  # Restrict the DataFrame to selected columns when requested
                dataframe = dataframe[columns]  # Filter the DataFrame to the desired columns
            if analysis_type == "summary":  # Produce descriptive statistics when a summary is requested
                result = {  # Build the summary result payload
                    "shape": dataframe.shape,  # Include the DataFrame shape
                    "columns": list(dataframe.columns),  # Include the list of columns
                    "dtypes": dataframe.dtypes.apply(lambda dtype: str(dtype)).to_dict(),  # Include column data types as strings
                    "summary": dataframe.describe(include="all").fillna("").to_dict(),  # Include descriptive statistics for each column
                    "missing_values": dataframe.isnull().sum().to_dict()  # Include the count of missing values per column
                }  # Close the summary result payload
            elif analysis_type == "correlation":  # Produce a correlation matrix when requested
                result = {"correlation_matrix": dataframe.corr(numeric_only=True).to_dict()}  # Include the numeric correlation matrix
            else:  # Handle unsupported analysis types gracefully
                result = {"error": f"Unsupported analysis type: {analysis_type}"}  # Inform the caller about the unsupported analysis type
            return json.dumps(result, indent=2)  # Serialize the analysis result as formatted JSON
        except Exception as error:  # Handle unexpected failures gracefully
            logger.error("Data analysis failed: %s", error)  # Log the failure details
            return f"Error: {error}"  # Return an error string to the caller


class ToolFactory:  # Provide helpers to construct collections of tools
    @staticmethod  # Indicate that the method does not rely on instance state
    def get_all_tools() -> List[BaseTool]:  # Return all available tool instances
        return [  # Return a list of instantiated tools
            WebSearchTool(),  # Include the web search tool
            FileReadTool(),  # Include the file reader tool
            FileWriteTool(),  # Include the file writer tool
            CodeExecuteTool(),  # Include the code execution tool
            APIRequestTool(),  # Include the API request tool
            DataAnalysisTool()  # Include the data analysis tool
        ]  # Close the list of tool instances

    @staticmethod  # Indicate that the method does not rely on instance state
    def get_tool_by_name(name: str) -> Optional[BaseTool]:  # Retrieve a tool instance by its human-friendly name
        for tool in ToolFactory.get_all_tools():  # Iterate through all available tools
            if tool.name.lower() == name.lower():  # Compare tool names case-insensitively
                return tool  # Return the first matching tool
        return None  # Return None when no matching tool is found

    @staticmethod  # Indicate that the method does not rely on instance state
    def get_tools_by_category(category: str) -> List[BaseTool]:  # Retrieve tools grouped by logical categories
        categories: Dict[str, List[BaseTool]] = {  # Define the available tool categories
            "web": [WebSearchTool()],  # Group web-related tools
            "file": [FileReadTool(), FileWriteTool()],  # Group file management tools
            "code": [CodeExecuteTool()],  # Group code execution tools
            "api": [APIRequestTool()],  # Group API interaction tools
            "data": [DataAnalysisTool()]  # Group data analysis tools
        }  # Close the category mapping
        return categories.get(category.lower(), [])  # Return the matching category or an empty list by default


class ToolRegistry:  # Provide a registry that caches tool instances for reuse
    def __init__(self) -> None:  # Initialize the registry with default tools
        self.tools: Dict[str, BaseTool] = {}  # Initialize the tool mapping keyed by human-friendly name
        self.load_default_tools()  # Populate the registry with default tool instances

    def load_default_tools(self) -> None:  # Load the default tool set into the registry
        for tool in ToolFactory.get_all_tools():  # Iterate through the factory-provided tools
            self.register_tool(tool)  # Register each tool with the registry

    def register_tool(self, tool: BaseTool) -> None:  # Register a tool instance under its name
        self.tools[tool.name] = tool  # Store the tool in the registry mapping
        logger.info("Registered tool %s", tool.name)  # Log the registration event

    def unregister_tool(self, tool_name: str) -> bool:  # Remove a tool from the registry by name
        if tool_name in self.tools:  # Check whether the tool exists in the registry
            del self.tools[tool_name]  # Remove the tool entry from the registry
            logger.info("Unregistered tool %s", tool_name)  # Log the removal event
            return True  # Indicate that the tool was removed
        logger.warning("Attempted to unregister missing tool %s", tool_name)  # Warn when the tool was not found
        return False  # Indicate that no tool was removed

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:  # Retrieve a registered tool by name
        return self.tools.get(tool_name)  # Return the tool instance or None when missing

    def list_tools(self) -> List[str]:  # List the names of all registered tools
        return list(self.tools.keys())  # Return the list of registered tool names

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:  # Retrieve metadata about a registered tool
        tool = self.get_tool(tool_name)  # Look up the tool in the registry
        if tool:  # Check whether the tool exists
            return {  # Return metadata describing the tool
                "name": tool.name,  # Include the tool name
                "description": tool.description,  # Include the tool description
                "args_schema": str(tool.args_schema)  # Include the string representation of the argument schema
            }  # Close the metadata dictionary
        return None  # Return None when the tool is not registered
