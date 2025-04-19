from pydantic import BaseModel, Field
from typing import Literal, Optional

class WeatherParams(BaseModel):
    location: str = Field(description="The location for which to get the weather forecast.")
    units: Literal["celsius", "fahrenheit"] = Field(
        description="The unit of temperature measurement. Can be 'celsius' or 'fahrenheit'."
    )

def get_weather(params: WeatherParams) -> dict:
    """
    Retrieves current weather for the given location.
    """
    # Placeholder for actual weather fetching logic
    return {
        "location": params.location,
        "temperature": 25,
        "units": params.units,
        "forecast": "Sunny"
    }