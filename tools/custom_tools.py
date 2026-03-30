from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
import requests
import json
import os
import logging
import subprocess
import re
import tempfile

logger = logging.getLogger(__name__)

class WebSearchInput(BaseModel):
    """Input schema for web search tool"""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    search_engine: str = Field(default="duckduckgo", description="Search engine to use")

class WebSearchTool(BaseTool):
    """Custom web search tool"""
    name: str = "Web Search"
    description: str = "Search the web for information"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 5, search_engine: str = "duckduckgo") -> str:
        """Execute web search"""
        try:
            # This is a mock implementation - replace with actual search API
            search_results = [
                {
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a sample result {i+1} for the search query: {query}"
                }
                for i in range(max_results)
            ]
            
            return json.dumps(search_results, indent=2)
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return f"Error: {str(e)}"

class FileReadInput(BaseModel):
    """Input schema for file read tool"""
    file_path: str = Field(..., description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")

_ALLOWED_READ_DIR = os.path.abspath(os.getenv("TOOL_DATA_DIR", "data"))


class FileReadTool(BaseTool):
    """Custom file reading tool"""
    name: str = "File Reader"
    description: str = "Read content from a file"
    args_schema: Type[BaseModel] = FileReadInput

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file content"""
        try:
            resolved = os.path.realpath(os.path.abspath(file_path))
            if not resolved.startswith(_ALLOWED_READ_DIR + os.sep) and resolved != _ALLOWED_READ_DIR:
                return f"Error: Access denied: path outside allowed directory"

            if not os.path.exists(resolved):
                return f"Error: File not found: {file_path}"

            with open(resolved, 'r', encoding=encoding) as f:
                content = f.read()

            return content

        except Exception as e:
            logger.error(f"File read failed: {str(e)}")
            return f"Error: {str(e)}"

class FileWriteInput(BaseModel):
    """Input schema for file write tool"""
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")

_ALLOWED_WRITE_DIR = os.path.abspath(os.getenv("TOOL_DATA_DIR", "data"))


class FileWriteTool(BaseTool):
    """Custom file writing tool"""
    name: str = "File Writer"
    description: str = "Write content to a file"
    args_schema: Type[BaseModel] = FileWriteInput

    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to file"""
        try:
            resolved = os.path.realpath(os.path.abspath(file_path))
            if not resolved.startswith(_ALLOWED_WRITE_DIR + os.sep) and resolved != _ALLOWED_WRITE_DIR:
                return f"Error: Access denied: path outside allowed directory"

            os.makedirs(os.path.dirname(resolved), exist_ok=True)

            with open(resolved, 'w', encoding=encoding) as f:
                f.write(content)

            return f"Successfully wrote to file: {file_path}"

        except Exception as e:
            logger.error(f"File write failed: {str(e)}")
            return f"Error: {str(e)}"

class CodeExecuteInput(BaseModel):
    """Input schema for code execution tool"""
    code: str = Field(..., description="Code to execute")
    language: str = Field(default="python", description="Programming language")
    timeout: int = Field(default=30, description="Execution timeout in seconds")

class CodeExecuteTool(BaseTool):
    """Custom code execution tool"""
    name: str = "Code Executor"
    description: str = "Execute code in a safe environment"
    args_schema: Type[BaseModel] = CodeExecuteInput
    
    def _run(self, code: str, language: str = "python", timeout: int = 30) -> str:
        """Execute code"""
        if language.lower() != "python":
            return "Error: Only Python is currently supported"

        temp_file = None
        try:
            # Use a securely-named temp file to avoid predictable paths
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=tempfile.gettempdir()
            ) as f:
                temp_file = f.name
                f.write(code)

            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"

        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)

class APIRequestInput(BaseModel):
    """Input schema for API request tool"""
    url: str = Field(..., description="API endpoint URL")
    method: str = Field(default="GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request data")
    timeout: int = Field(default=30, description="Request timeout in seconds")

class APIRequestTool(BaseTool):
    """Custom API request tool"""
    name: str = "API Request"
    description: str = "Make HTTP requests to APIs"
    args_schema: Type[BaseModel] = APIRequestInput
    
    def _run(self, url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, 
             data: Optional[Dict[str, Any]] = None, timeout: int = 30) -> str:
        """Make API request"""
        try:
            method = method.upper()
            
            request_kwargs = {
                "url": url,
                "timeout": timeout
            }
            
            if headers:
                request_kwargs["headers"] = headers
            
            if data and method in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data
            
            response = requests.request(method, **request_kwargs)
            
            return json.dumps({
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text
            }, indent=2)
            
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return f"Error: {str(e)}"

class DataAnalysisInput(BaseModel):
    """Input schema for data analysis tool"""
    data: str = Field(..., description="Data to analyze (JSON string)")
    analysis_type: str = Field(default="summary", description="Type of analysis")
    columns: Optional[List[str]] = Field(default=None, description="Columns to analyze")

class DataAnalysisTool(BaseTool):
    """Custom data analysis tool"""
    name: str = "Data Analyzer"
    description: str = "Analyze structured data"
    args_schema: Type[BaseModel] = DataAnalysisInput
    
    def _run(self, data: str, analysis_type: str = "summary", columns: Optional[List[str]] = None) -> str:
        """Analyze data"""
        try:
            import pandas as pd
            
            # Parse JSON data
            data_dict = json.loads(data)
            df = pd.DataFrame(data_dict)
            
            if columns:
                df = df[columns]
            
            if analysis_type == "summary":
                result = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "summary": df.describe().to_dict(),
                    "missing_values": df.isnull().sum().to_dict()
                }
            elif analysis_type == "correlation":
                result = {
                    "correlation_matrix": df.corr().to_dict()
                }
            else:
                result = {"error": f"Unsupported analysis type: {analysis_type}"}
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return f"Error: {str(e)}"

class ToolFactory:
    """Factory class for creating tools"""
    
    @staticmethod
    def get_all_tools() -> List[BaseTool]:
        """Get all available tools"""
        return [
            WebSearchTool(),
            FileReadTool(),
            FileWriteTool(),
            CodeExecuteTool(),
            APIRequestTool(),
            DataAnalysisTool()
        ]
    
    @staticmethod
    def get_tool_by_name(name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        tools = ToolFactory.get_all_tools()
        
        for tool in tools:
            if tool.name.lower() == name.lower():
                return tool
        
        return None
    
    @staticmethod
    def get_tools_by_category(category: str) -> List[BaseTool]:
        """Get tools by category"""
        categories = {
            "web": [WebSearchTool()],
            "file": [FileReadTool(), FileWriteTool()],
            "code": [CodeExecuteTool()],
            "api": [APIRequestTool()],
            "data": [DataAnalysisTool()]
        }
        
        return categories.get(category.lower(), [])

class ToolRegistry:
    """Registry for managing custom tools"""
    
    def __init__(self):
        self.tools = {}
        self.load_default_tools()
    
    def load_default_tools(self):
        """Load default tools"""
        default_tools = ToolFactory.get_all_tools()
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        
        logger.warning(f"Tool not found: {tool_name}")
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a registered tool"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information"""
        tool = self.get_tool(tool_name)
        
        if tool:
            return {
                "name": tool.name,
                "description": tool.description,
                "args_schema": str(tool.args_schema)
            }
        
        return None
