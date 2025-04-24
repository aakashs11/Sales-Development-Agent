from openai import OpenAI
client = OpenAI()
import pandas as pd
from agent.agent import Agent
import logging
from tools.query_dataframe import QueryDataFrameTool
from utils.function_to_schemas import function_to_schema
from tools.query_dataframe import QueryDataFrameInput, QueryDataFrameOutput
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#Initialize query tool
df = pd.read_csv(r"cleaned_leads.csv")
QUERY_DATAFRAME_TOOL_INSTRUCTIONS = open(r"prompts\query_dataframe_tool\prompt_v1.txt", encoding="utf-8").read()
query_tool = QueryDataFrameTool(df, QUERY_DATAFRAME_TOOL_INSTRUCTIONS)

#Tool definition
tools = {"query_dataframe": query_tool.query_dataframe}
tool_models = {"query_dataframe": QueryDataFrameInput}
tool_schemas = [function_to_schema(query_tool.query_dataframe, QueryDataFrameInput)]

#Agent initialization
SDR_ASSISTANT_INSTRUCTIONS = open(r"prompts\sdr_agent\prompt_v1.txt", encoding="utf-8").read()
agent = Agent("SalesAgent", client, SDR_ASSISTANT_INSTRUCTIONS, tool_schemas, tool_models, tools)
agent_response = agent.response("can you show leads which are not converted from these leads?")


output_schema ={
    "response": "Here are the leads not converted...",
    "thoughts": [],
    "actions": [],
    "tool_output": df.to_dict(orient="records"),  # or a summary/preview
    "state": {
        "current_agent": "SalesAgent",
        "last_tool_used": "query_dataframe"
    },
    "metadata": {
        "timestamp": "2025-04-24T12:34:56",
        "agent_version": "v1.2"
    },
    "error": None
}