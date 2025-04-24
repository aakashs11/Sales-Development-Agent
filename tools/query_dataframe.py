
import json
import numpy as np
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI


class QueryDataFrameInput(BaseModel):
    user_query: str = Field(description="The natural language query to be executed on the DataFrame.")

class QueryDataFrameOutput(BaseModel):
    status: str
    explanation: Optional[str] = None
    data: Optional[Union[List[Dict[str, Any]], str, float, int]] = None
    error: Optional[str] = None

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
    def __init__(self, df: pd.DataFrame, developer_instructions: str):
        self.df = df
        self.prompt_template = developer_instructions 
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

    def update_dataframe(self, new_df: pd.DataFrame):
        """Update the internal DataFrame."""
        self.df = new_df

    def run_query(self, expr: str) -> pd.DataFrame:
        """
        Run an arbitrary pandas expression on the DataFrame using eval.
        WARNING: This is not safe for untrusted input!
        """
        print("Running query:", expr)
        try:
            local_vars = {"df": self.df}
            safe_builtins = {"len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round}

            result = eval(expr, {"__builtins__": safe_builtins}, local_vars)

            if isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, (pd.Series, list, dict)):
                return pd.DataFrame(result)
            else:
                # Wrap scalar results in a DataFrame
                return pd.DataFrame([{"result": result}])
            
        except Exception as e:
            raise ValueError(f"Query failed: {e}")

    def parse_llm_output(self, llm_response: str) -> dict:
        """Parse the LLM output string into a structured dict according to RESPONSE_SCHEMA."""     
        try:
            parsed = json.loads(llm_response)
            return parsed
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")

    def parse_pandasai_output(self, output) -> QueryDataFrameOutput:
        """Normalize PandasAI output to QueryDataFrameOutput."""
        if isinstance(output, pd.DataFrame):
            return QueryDataFrameOutput(
                status="success",
                explanation="PandasAI returned a DataFrame.",
                data=output.to_dict(orient="records"),
                error=None
            )
        elif isinstance(output, (int, float, str, bool, np.integer, np.floating)):
            if isinstance(output, (np.integer, np.floating)):
                output = output.item()
            return QueryDataFrameOutput(
                status="success",
                explanation=f"PandasAI returned a scalar: {output}",
                data=output,
                error=None
            )
        elif isinstance(output, dict) and "value" in output:
            return QueryDataFrameOutput(
                status="success",
                explanation=f"PandasAI returned a {output.get('type', 'value')}.",
                data=output["value"],
                error=None
            )
        elif isinstance(output, (list, tuple)):
            return QueryDataFrameOutput(
                status="success",
                explanation="PandasAI returned a list/tuple.",
                data=output,
                error=None
            )
        elif hasattr(output, "savefig"):  # Matplotlib Figure
            output.savefig("temp_chart.png")
            return QueryDataFrameOutput(
                status="success",
                explanation="PandasAI returned a chart.",
                data="temp_chart.png",
                error=None
            )
        else:
            return QueryDataFrameOutput(
                status="error",
                explanation=f"Unsupported output type: {type(output)}",
                data=None,
                error=str(output)
            )

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
        max_retries = 2
        attempt = 0  
        last_error = None 

        while attempt < max_retries:
            try:
                schema_description = self.extract_dataframe_schema(self.df)
                prompt = self.prompt_template.format(
                    schema_description=schema_description,
                    user_request=user_query
                )
                response = self.client.responses.create(
                    model="gpt-4o",
                    input=prompt,
                    text=self.response_schema,
                )
                print("LLM response:", response)
                llm_result = self.parse_llm_output(response.output_text)
                print("Parsed LLM result:", llm_result)
                
                if not llm_result.get("output"):

                    raise ValueError("LLM did not return a valid output.")
                explanation = llm_result.get("explanation", "")
                query_expr = llm_result.get("output", "")
                query_result = self.run_query(query_expr)
                return QueryDataFrameOutput(
                    status="success",
                    explanation= explanation,
                    data= query_result.to_dict(orient="records"),
                    error= None
                )

            except Exception as e:
                last_error = str(e)
                attempt += 1
        return QueryDataFrameOutput(
            status="error",
            explanation= "",
            data= None,
            error=last_error if last_error else "Unknown error"
        )

    def secondary_pandasai_query(self, user_query: QueryDataFrameInput) -> QueryDataFrameOutput:
        """A fallback approach using PandasAI if the primary approach fails."""

        try:
            response = self.smart_df.chat(user_query)
            return self.parse_pandasai_output(response)
        except Exception as e:
            return QueryDataFrameOutput(
                status="error",
                explanation="PandasAI error",
                data=None,
                error=str(e)
            )
    
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
                print("Query result:", result)
                is_empty = (
                result.status != "success" or
                result.data is None or
                result.data == [] or
                result.data == {} or
                result.data == "" or
                (isinstance(result.data, pd.DataFrame) and result.data.empty)
                )
                print("Is empty:", is_empty)
                if not is_empty:
                    return result.data
                else:
                    last_error = result.error or "Empty result"
            except Exception as e:
                last_error = str(e)
                continue
        return QueryDataFrameOutput(
            status="error",
            explanation= "All attempts failed",
            data= None,
            error= last_error
        )
