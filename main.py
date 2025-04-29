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
#create a timestamp for the metadata
import datetime
timestamp = datetime.datetime.now().isoformat()

def run(input_text, memory):
    
    output_schema ={

        "response": "",
        "thoughts": [],
        "actions": [],
        "tool_output": "",  # or a summary/preview
        "state": {
            "current_agent": "SalesAgent",
        },
        "metadata": {
            "timestamp": timestamp,
            "agent_version": "v1.2"
        },
        "error": None
    }
    
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
    agent = Agent("SalesAgent", client, SDR_ASSISTANT_INSTRUCTIONS, tool_schemas, tool_models, tools, memory=memory)
    agent_response = agent.response(input_text)

    response = agent_response.get("response", "No response generated.")
    memory = agent_response.get("memory", [])
    thoughts = agent_response.get("thoughts", [])
    actions = agent_response.get("actions", [])
    tool_output = agent_response.get("tool_output", "No tool output generated.")
    state = agent_response.get("state", {})
    metadata = agent_response.get("metadata", {})
    error = agent_response.get("error", None)

    # Log the response and other details
    logging.info("Response: %s", response)
    logging.info("memory: %s", agent_response.get("memory", []))
    logging.info("Thoughts: %s", thoughts)
    logging.info("Actions: %s", actions)
    logging.info("Tool Output: %s", tool_output)
    logging.info("State: %s", state)
    logging.info("Metadata: %s", metadata)
    logging.info("Error: %s", error)

    # Return the final response
    # Note: You might want to format the response as needed
    # For example, you can return a dictionary or a string
    # depending on your application's requirements.
    # Here, we are returning a dictionary with the response and other details
    return {
        "response": response,
        "memory": memory,
        "thoughts": thoughts,
        "actions": actions,
        "tool_output": tool_output,
        "state": state,
        "metadata": metadata,
        "error": error,
        
    }

