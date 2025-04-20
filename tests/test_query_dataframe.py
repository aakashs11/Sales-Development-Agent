import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from tools.query_dataframe import QueryDataFrameTool, QueryDataFrameInput
prompt_path = r"prompts\query_dataframe_tool\prompt_v1.txt"

def test_extract_dataframe_schema():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tool = QueryDataFrameTool(df, prompt_path=prompt_path)
    schema = tool.extract_dataframe_schema(df)
    assert "a" in schema and "b" in schema

def test_run_query_success():
    df = pd.DataFrame({"a": [1, 2, 3]})
    tool = QueryDataFrameTool(df, prompt_path=prompt_path)
    result = tool.run_query("a > 1")
    assert not result.empty

def test_run_query_failure():
    df = pd.DataFrame({"a": [1, 2, 3]})
    tool = QueryDataFrameTool(df, prompt_path=prompt_path)
    with pytest.raises(ValueError):
        tool.run_query("invalid >")

def test_parse_llm_output_valid():
    tool = QueryDataFrameTool(pd.DataFrame(), prompt_path=prompt_path)
    output = tool.parse_llm_output('{"explanation": "ok", "output": "a > 1"}')
    assert output["explanation"] == "ok"

def test_parse_llm_output_invalid():
    tool = QueryDataFrameTool(pd.DataFrame(), prompt_path=prompt_path)
    with pytest.raises(ValueError):
        tool.parse_llm_output('not a json')

# For primary_llm_query, secondary_pandasai_query, and query_dataframe,
# use mocking to simulate LLM and PandasAI responses.