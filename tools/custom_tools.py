from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
import requests
import json
import os
import logging
import subprocess
from pathlib import Path
import tempfile
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_BASE_WORKSPACE = Path(os.getenv("AGENT_TOOLS_BASE_DIR", "/workspace")).expanduser().resolve()
try:
    _BASE_WORKSPACE.mkdir(parents=True, exist_ok=True)
except Exception as exc:
    logger.warning("Unable to ensure workspace directory exists: %s", exc)


def _resolve_within_workspace(path: str) -> Path:
    """Ensure the requested path stays within the allowed workspace."""
    candidate = Path(path).expanduser().resolve()
    try:
        candidate.relative_to(_BASE_WORKSPACE)
    except ValueError as exc:
        raise ValueError(
            f"Path '{candidate}' is outside the permitted workspace: {_BASE_WORKSPACE}"
        ) from exc
    return candidate


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

class FileReadTool(BaseTool):
    """Custom file reading tool"""
    name: str = "File Reader"
    description: str = "Read content from a file"
    args_schema: Type[BaseModel] = FileReadInput
    
    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        """Read file content"""
        try:
            safe_path = _resolve_within_workspace(file_path)
            if not safe_path.exists():
                return f"Error: File not found: {safe_path}"
            
            with safe_path.open('r', encoding=encoding) as f:
                return f.read()
            
        except ValueError as exc:
            logger.warning("Blocked file read: %s", exc)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("File read failed (%s): %s", file_path, exc)
            return f"Error: {str(exc)}"

class FileWriteInput(BaseModel):
    """Input schema for file write tool"""
    file_path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write")
    encoding: str = Field(default="utf-8", description="File encoding")

class FileWriteTool(BaseTool):
    """Custom file writing tool"""
    name: str = "File Writer"
    description: str = "Write content to a file"
    args_schema: Type[BaseModel] = FileWriteInput
    
    def _run(self, file_path: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to file"""
        try:
            safe_path = _resolve_within_workspace(file_path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            
            with safe_path.open('w', encoding=encoding) as f:
                f.write(content)
            
            return f"Successfully wrote to file: {safe_path}"
            
        except ValueError as exc:
            logger.warning("Blocked file write: %s", exc)
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("File write failed (%s): %s", file_path, exc)
            return f"Error: {str(exc)}"

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
        temp_file: Optional[str] = None
        try:
            if language.lower() != "python":
                return "Error: Only Python is currently supported"
            
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                temp_file = tmp.name
            
            result = subprocess.run(
                ["python", "-I", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_BASE_WORKSPACE),
            )
            
            if result.returncode == 0:
                return result.stdout
            return f"Error: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as exc:
            logger.error("Code execution failed: %s", exc)
            return f"Error: {str(exc)}"
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as cleanup_exc:
                    logger.warning("Failed to remove temp file %s: %s", temp_file, cleanup_exc)

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
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                return "Error: Only HTTP/HTTPS URLs are supported"
            
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
            
        except requests.RequestException as exc:
            logger.error("API request failed: %s", exc)
            return f"Error: {str(exc)}"
        except ValueError as exc:
            return f"Error: {exc}"

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
