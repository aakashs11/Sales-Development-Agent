
def function_to_schema(func,params_model):
    """
    Convert a function and its parameters model to a schema dictionary.
    
    Args:
        func: The function to convert.
        params_model: The Pydantic model for the function's parameters.
        
    Returns:
        A dictionary representing the function schema.
    """
    schema = params_model.model_json_schema()
    schema["additionalProperties"] = False 

    return {
        "type": "function",
        "name": func.__name__.strip(),
        "description": func.__doc__ or "",
        "strict": True,
        "parameters": schema,  

    }
