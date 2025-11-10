from typing import Any, Dict, List, Optional, Type  # Provide typing helpers for clarity and static analysis
import json  # Provide JSON serialization utilities for consistent payloads
import logging  # Provide logging utilities for observability within tool abstractions
import os  # Provide filesystem helpers for secure file manipulation
import subprocess  # Provide subprocess utilities for controlled code execution
from datetime import datetime  # Provide timestamp utilities for audit trails

import requests  # Provide HTTP client capabilities for API requests
from crewai_tools import BaseTool  # Provide base class for custom CrewAI tools
from pydantic import BaseModel, Field  # Provide Pydantic models for request validation

logger = logging.getLogger(__name__)  # Acquire module-specific logger for contextualized diagnostics


class WebSearchInput(BaseModel):  # Define validated input schema for web search tool
    query: str = Field(..., description="Search query string to submit")  # Ensure query is provided for search execution
    max_results: int = Field(default=5, ge=1, le=25, description="Maximum number of results to return")  # Constrain result count for safety
    search_engine: str = Field(default="duckduckgo", description="Search engine identifier")  # Allow explicit engine selection


class WebSearchTool(BaseTool):  # Define custom tool to perform web search
    name: str = "Web Search"  # Provide human-readable tool name
    description: str = "Search the web for information"  # Provide tool description for documentation
    args_schema: Type[BaseModel] = WebSearchInput  # Associate validated input schema with tool

    def _run(self, query: str, max_results: int = 5, search_engine: str = "duckduckgo") -> str:  # Define synchronous execution method
        try:  # Provide error handling for search execution
            results = [  # Construct deterministic mocked response to avoid outbound calls by default
                {
                    "title": f"Result {index + 1} for {query}",  # Provide synthesized result title
                    "url": f"https://example.com/result{index + 1}",  # Provide synthesized result URL
                    "snippet": f"Synthetic result {index + 1} generated for query '{query}'.",  # Provide synthesized snippet
                }
                for index in range(max_results)  # Iterate desired result count
            ]  # Close comprehension
            return json.dumps(results, indent=2)  # Serialize results to JSON string with indentation
        except Exception as exc:  # Catch unexpected failures
            logger.exception("Web search execution failed: %s", exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller


class FileReadInput(BaseModel):  # Define validated input schema for file read tool
    file_path: str = Field(..., description="Absolute path to file for reading")  # Ensure file path provided
    encoding: str = Field(default="utf-8", description="Text encoding used when reading file")  # Allow encoding override


class FileReadTool(BaseTool):  # Define custom tool to read file content
    name: str = "File Reader"  # Provide human-readable tool name
    description: str = "Read content from a file"  # Provide tool description for documentation
    args_schema: Type[BaseModel] = FileReadInput  # Associate validated input schema with tool

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:  # Define synchronous execution method
        try:  # Provide error handling for file read
            if not os.path.exists(file_path):  # Verify file exists before attempting to read
                return f"Error: File not found: {file_path}"  # Return error when file missing
            with open(file_path, "r", encoding=encoding) as handle:  # Open file safely using context manager
                content = handle.read()  # Read entire file contents
            return content  # Return file contents to caller
        except Exception as exc:  # Catch unexpected failures
            logger.exception("File read failed for %s: %s", file_path, exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller


class FileWriteInput(BaseModel):  # Define validated input schema for file write tool
    file_path: str = Field(..., description="Absolute path to file for writing")  # Ensure file path provided
    content: str = Field(..., description="Content to persist to file")  # Ensure content provided
    encoding: str = Field(default="utf-8", description="Text encoding used when writing file")  # Allow encoding override


class FileWriteTool(BaseTool):  # Define custom tool to write content to file
    name: str = "File Writer"  # Provide human-readable tool name
    description: str = "Write content to a file"  # Provide tool description for documentation
    args_schema: Type[BaseModel] = FileWriteInput  # Associate validated input schema with tool

    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:  # Define synchronous execution method
        try:  # Provide error handling for file write
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)  # Ensure target directory exists securely
            with open(file_path, "w", encoding=encoding) as handle:  # Open file safely using context manager
                handle.write(content)  # Write provided content to file
            return f"Successfully wrote to file: {file_path}"  # Return success message to caller
        except Exception as exc:  # Catch unexpected failures
            logger.exception("File write failed for %s: %s", file_path, exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller


class CodeExecuteInput(BaseModel):  # Define validated input schema for code execution tool
    code: str = Field(..., description="Code snippet to execute")  # Ensure code provided
    language: str = Field(default="python", description="Programming language identifier")  # Restrict to supported language
    timeout: int = Field(default=30, ge=1, le=120, description="Execution timeout in seconds")  # Bound execution timeout for safety


class CodeExecuteTool(BaseTool):  # Define custom tool to execute code snippets
    name: str = "Code Executor"  # Provide human-readable tool name
    description: str = "Execute code in a sandboxed environment"  # Provide tool description emphasizing safety
    args_schema: Type[BaseModel] = CodeExecuteInput  # Associate validated input schema with tool

    def _run(self, code: str, language: str = "python", timeout: int = 30) -> str:  # Define synchronous execution method
        if language.lower() != "python":  # Enforce supported language constraint
            return "Error: Only Python execution is supported currently"  # Return error for unsupported languages
        temp_file = f"/tmp/code_execute_{datetime.now().timestamp()}.py"  # Construct unique temp file path for execution
        try:  # Provide error handling for code execution
            with open(temp_file, "w", encoding="utf-8") as handle:  # Write code to temp file securely
                handle.write(code)  # Persist code to file
            result = subprocess.run(  # Execute code using subprocess in controlled manner
                ["python", temp_file],  # Invoke Python interpreter on temp file
                capture_output=True,  # Capture stdout and stderr for return
                text=True,  # Decode outputs as text
                timeout=timeout,  # Enforce execution timeout to mitigate abuse
                check=False,  # Avoid raising exceptions to handle manually
            )  # Close subprocess invocation
            if result.returncode == 0:  # Determine success via return code
                return result.stdout or "Execution completed with no output"  # Return stdout or default message
            return f"Error: {result.stderr}"  # Return stderr when execution fails
        except subprocess.TimeoutExpired:  # Handle timeout explicitly
            return f"Error: Code execution timed out after {timeout} seconds"  # Return timeout message to caller
        except Exception as exc:  # Catch unexpected failures
            logger.exception("Code execution failed: %s", exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller
        finally:  # Ensure cleanup executes regardless of outcome
            if os.path.exists(temp_file):  # Check whether temp file still exists
                os.remove(temp_file)  # Remove temp file to prevent residue


class APIRequestInput(BaseModel):  # Define validated input schema for API request tool
    url: str = Field(..., description="Target API endpoint URL")  # Ensure URL provided
    method: str = Field(default="GET", description="HTTP method to use")  # Allow method override
    headers: Optional[Dict[str, str]] = Field(default=None, description="Optional request headers")  # Allow header injection
    data: Optional[Dict[str, Any]] = Field(default=None, description="Optional JSON payload")  # Allow JSON payload
    timeout: int = Field(default=30, ge=1, le=60, description="Request timeout in seconds")  # Bound timeout for safety


class APIRequestTool(BaseTool):  # Define custom tool to perform HTTP API requests
    name: str = "API Request"  # Provide human-readable tool name
    description: str = "Make HTTP requests to APIs"  # Provide tool description for documentation
    args_schema: Type[BaseModel] = APIRequestInput  # Associate validated input schema with tool

    def _run(  # Define synchronous execution method with explicit parameters
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ) -> str:
        try:  # Provide error handling for HTTP request
            normalized_method = method.upper()  # Normalize method name to uppercase
            request_kwargs: Dict[str, Any] = {"url": url, "timeout": timeout}  # Initialize request arguments
            if headers:  # Include headers when provided
                request_kwargs["headers"] = headers  # Attach headers to request arguments
            if data and normalized_method in {"POST", "PUT", "PATCH"}:  # Include payload when allowed by method
                request_kwargs["json"] = data  # Attach JSON payload securely
            response = requests.request(normalized_method, **request_kwargs)  # Execute HTTP request using requests library
            payload = {  # Construct response payload for caller
                "status_code": response.status_code,  # Include HTTP status code
                "headers": dict(response.headers),  # Include response headers
                "content": response.text,  # Include response body as text
            }
            return json.dumps(payload, indent=2)  # Serialize payload to JSON string with indentation
        except Exception as exc:  # Catch unexpected failures
            logger.exception("API request to %s failed: %s", url, exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller


class DataAnalysisInput(BaseModel):  # Define validated input schema for data analysis tool
    data: str = Field(..., description="JSON string representing tabular data")  # Ensure JSON data provided
    analysis_type: str = Field(default="summary", description="Type of analysis to perform")  # Allow analysis selection
    columns: Optional[List[str]] = Field(default=None, description="Subset of columns to analyze")  # Allow column filtering


class DataAnalysisTool(BaseTool):  # Define custom tool to perform basic data analysis
    name: str = "Data Analyzer"  # Provide human-readable tool name
    description: str = "Analyze structured data"  # Provide tool description for documentation
    args_schema: Type[BaseModel] = DataAnalysisInput  # Associate validated input schema with tool

    def _run(self, data: str, analysis_type: str = "summary", columns: Optional[List[str]] = None) -> str:  # Define synchronous execution method
        try:  # Provide error handling for analysis
            import pandas as pd  # Import pandas lazily to avoid unnecessary dependency cost

            parsed = json.loads(data)  # Parse JSON string into Python object
            dataframe = pd.DataFrame(parsed)  # Construct DataFrame from parsed data
            if columns:  # Apply column filtering when requested
                dataframe = dataframe[columns]  # Select subset of columns safely
            if analysis_type == "summary":  # Handle summary analysis
                result = {  # Construct summary payload
                    "shape": dataframe.shape,  # Include DataFrame shape
                    "columns": list(dataframe.columns),  # Include column names
                    "dtypes": dataframe.dtypes.astype(str).to_dict(),  # Include column data types as strings
                    "summary": dataframe.describe(include="all").to_dict(),  # Include descriptive statistics
                    "missing_values": dataframe.isnull().sum().to_dict(),  # Include missing value counts
                }
            elif analysis_type == "correlation":  # Handle correlation analysis
                result = {"correlation_matrix": dataframe.corr(numeric_only=True).to_dict()}  # Include correlation matrix
            else:  # Handle unsupported analysis type gracefully
                result = {"error": f"Unsupported analysis type: {analysis_type}"}  # Provide helpful error payload
            return json.dumps(result, indent=2, default=str)  # Serialize result to JSON string with indentation
        except Exception as exc:  # Catch unexpected failures
            logger.exception("Data analysis failed: %s", exc)  # Emit exception log capturing context
            return f"Error: {exc}"  # Return error message to caller


class ToolFactory:  # Provide factory helpers for instantiating tools
    @staticmethod  # Declare method does not use instance or class state
    def get_all_tools() -> List[BaseTool]:  # Return collection of all supported tools
        return [  # Return list of instantiated tools
            WebSearchTool(),  # Include web search tool
            FileReadTool(),  # Include file read tool
            FileWriteTool(),  # Include file write tool
            CodeExecuteTool(),  # Include code execute tool
            APIRequestTool(),  # Include API request tool
            DataAnalysisTool(),  # Include data analysis tool
        ]

    @staticmethod  # Declare method does not use instance or class state
    def get_tool_by_name(name: str) -> Optional[BaseTool]:  # Retrieve tool by case-insensitive name
        for tool in ToolFactory.get_all_tools():  # Iterate through available tools
            if tool.name.lower() == name.lower():  # Compare names ignoring case
                return tool  # Return matching tool instance
        return None  # Return None when no match found

    @staticmethod  # Declare method does not use instance or class state
    def get_tools_by_category(category: str) -> List[BaseTool]:  # Retrieve tools grouped by category
        categories: Dict[str, List[BaseTool]] = {  # Define mapping of category names to tool instances
            "web": [WebSearchTool()],  # Provide web category tools
            "file": [FileReadTool(), FileWriteTool()],  # Provide file category tools
            "code": [CodeExecuteTool()],  # Provide code category tools
            "api": [APIRequestTool()],  # Provide API category tools
            "data": [DataAnalysisTool()],  # Provide data category tools
        }
        return categories.get(category.lower(), [])  # Return matching category or empty list


class ToolRegistry:  # Provide registry for managing custom tool lifecycles
    def __init__(self) -> None:  # Initialize registry instance
        self.tools: Dict[str, BaseTool] = {}  # Initialize internal dictionary to store tools by name
        self.load_default_tools()  # Preload default tools upon instantiation

    def load_default_tools(self) -> None:  # Load default tool set into registry
        for tool in ToolFactory.get_all_tools():  # Iterate through all default tools
            self.register_tool(tool)  # Register each tool with registry

    def register_tool(self, tool: BaseTool) -> None:  # Register new tool instance securely
        self.tools[tool.name] = tool  # Store tool instance keyed by name
        logger.info("Registered tool: %s", tool.name)  # Emit informational log capturing registration event

    def unregister_tool(self, tool_name: str) -> bool:  # Remove registered tool by name
        if tool_name in self.tools:  # Check whether tool exists
            del self.tools[tool_name]  # Remove tool from registry
            logger.info("Unregistered tool: %s", tool_name)  # Emit informational log capturing removal event
            return True  # Indicate removal success
        logger.warning("Tool %s not found during unregister attempt", tool_name)  # Emit warning when tool absent
        return False  # Indicate removal failure

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:  # Retrieve registered tool by name
        return self.tools.get(tool_name)  # Return tool instance or None

    def list_tools(self) -> List[str]:  # List names of all registered tools
        return list(self.tools.keys())  # Return list of tool names

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:  # Retrieve metadata about registered tool
        tool = self.get_tool(tool_name)  # Fetch tool instance by name
        if tool:  # Provide metadata when tool exists
            return {
                "name": tool.name,  # Include tool name
                "description": tool.description,  # Include tool description
                "args_schema": str(tool.args_schema),  # Include schema representation for transparency
            }
        return None  # Return None when tool not found
