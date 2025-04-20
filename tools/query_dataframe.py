
import json
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI
from pydantic import BaseModel
from typing import Any, Dict, List, Union
from openai import OpenAI


class QueryDataFrameInput(BaseModel):
    nl_query: str

class QueryDataFrameOutput(BaseModel):
    result: Union[List[Dict[str, Any]], str]

class QueryDataFrameTool:
    """A tool for querying a pandas DataFrame using natural language queries."""
    RESPONSE_SCHEMA ={
            "format": {
                "type": "json_schema",
                "name": "response_json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                                "explanation": { "type": "string" },
                                "output": { "type": "string" },
                    },
                "required": ["explanation", "output"],
                "additionalProperties": False
                },
                "strict": True
            }
        }
    def __init__(self, df: pd.DataFrame, prompt_path: str):
        self.df = df
        self.prompt_template = self.load_prompt(prompt_path) 
        self.response_schema = self.RESPONSE_SCHEMA
        self.client = OpenAI()
        self.llm_pandasai = PandasAI_OpenAI(model="gpt-4o", temperature=0)
        self.smart_df = SmartDataframe(
            self.df,
            config={
                "llm": self.llm_pandasai,
                "custom_prompts": {
                    "pandas_code": f" You are analyzing a sales leads dataset. Columns include {self.df.columns.tolist()} Be careful with column names and types.",
                },
                "verbose": True,
            },
        )
        
    
    def load_prompt(self, path: str) -> str:
        """Load the prompt template from a file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def update_dataframe(self, new_df: pd.DataFrame):
        """Update the internal DataFrame."""
        self.df = new_df

    def run_query(self, filter_expr: str) -> pd.DataFrame:
        """Run a query on the DataFrame using the provided filter expression."""
        try:
            return self.df.query(filter_expr)
        except Exception as e:
            raise ValueError(f"Query failed: {e}")

    def parse_llm_output(self, llm_response: str) -> dict:
        """Parse the LLM output string into a structured dict according to RESPONSE_SCHEMA."""     
        try:
            parsed = json.loads(llm_response)
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")

    def extract_dataframe_schema(self, df: pd.DataFrame, columns: list[str] = None) -> dict:
        """Returns a schema dict with column names, types, and optionally sample values. """
        if columns is None:
            columns = df.columns.tolist()
        schema = {}
        for col in columns:
            schema[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().unique()[:3].tolist()  # up to 3 unique values
            }
        return schema

    def primary_llm_query(self, user_query: QueryDataFrameInput) -> QueryDataFrameOutput:
        """
        Sends the natural language request to an LLM to generate a pandas query,
        executes it, and returns the result. Attempts multiple retries; if it fails,
        tries fallback with PandasAI.
        """
        # Here you would call your LLM or query logic
        # For demonstration, just return the columns
        max_retries = 3
        attempt = 0  

        while attempt < max_retries:
            try:
                schema_description = self.extract_dataframe_schema(self.df)
                prompt = self.prompt_template.format(
                    schema_description=schema_description,
                    user_request=user_query
                )
                response = self.client.responses.create(
                    model="gpt-4.1",
                    input=prompt,
                    text=self.response_schema,
                )
                llm_result = self.parse_llm_output(response.output_text)
                explanation = llm_result.get("explanation", "")
                query_expr = llm_result.get("output", "")
                query_result = self.run_query(query_expr)
                return QueryDataFrameOutput(result={
                    "status": "success",
                    "explanation": explanation,
                    "data": query_result.to_dict(orient="records"),
                    "error": None
                })

            except Exception as e:
                attempt += 1
                return QueryDataFrameOutput(result={
                    "status": "error",
                    "explanation": "",
                    "data": query_result.to_dict(orient="records"),
                    "error": str(e)
                })

    def secondary_pandasai_query(self, user_query: QueryDataFrameInput) -> QueryDataFrameOutput:
        """A fallback approach using PandasAI if the primary approach fails."""

        try:
            response = self.smart_df.chat(user_query)
            if isinstance(response, pd.DataFrame):
                return QueryDataFrameOutput(result={
                    "status": "success",
                    "explanation": "PandasAI response",
                    "data": response.to_dict(orient="records"),
                     "error": None
                })
            else:
                return QueryDataFrameOutput(result={
                    "explanation": "PandasAI response",
                    "data": response,
                     "error": None
                })
            
        except Exception as e:
            return QueryDataFrameOutput(result={
            "status": "success",
            "explanation": "PandasAI error",
            "data": str(e)
        })
    
    def query_dataframe(self, user_query: QueryDataFrameInput) -> QueryDataFrameOutput:
        """Main method to query the DataFrame."""
        fallbacks = [
            self.primary_llm_query,
            self.secondary_pandasai_query
        ]
        last_error = None
        for fallback in fallbacks:
            try:
                result = fallback(user_query)
                if result["status"] == "success":
                    return result
                else:
                    last_error = result["error"]
                return fallback(user_query)
            except Exception as e:
                last_error = str(e)
                continue
        return QueryDataFrameOutput(result={
            "status": "error",
            "explanation": "All attempts failed",
            "data": None,
            "error": last_error
        })
