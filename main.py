from openai import OpenAI
client = OpenAI()
from agent import Agent
import logging
from tools.get_weather import get_weather, WeatherParams
from utils.function_to_schemas import function_to_schema
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'
)


system_prompt = "You are a helpful agent"
tools = {"get_weather": get_weather}
tool_schemas = [function_to_schema(get_weather, WeatherParams)]
tool_models = {"get_weather": WeatherParams}
agent = Agent("SalesAgent", client, system_prompt, tool_schemas, tool_models, tools)