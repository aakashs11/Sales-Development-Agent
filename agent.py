import logging
import json
class Agent:
    def __init__(self, name, client, developer_instructions, tool_schemas=None, tool_models=None, tools=None):
        self.name = name
        self.client = client
        self.memory = [{"role": "developer", "content": developer_instructions}]
        self.tools = tools if tools is not None else {}  
        self.tool_schemas = tool_schemas if tool_schemas is not None else []
        self.tool_models = tool_models if tool_models is not None else {}

    def response(self, input_data):
        # Generate a response based on input data
        self.memory.append({"role": "user", "content": input_data})
        api_response_1= self.client.responses.create(
            model="gpt-4.1",
            input=self.memory,
            tools=self.tool_schemas,
            tool_choice="auto",
        )
        # logging.debug("API response: %s", api_response_1)

        # Process the API response
        for block in api_response_1.output:
            logging.debug("Block type: %s", block.type)

            if block.type == "message":

                logging.debug("Assistant message: %s", block.content[0].text)
                response_text = api_response_1.output_text
                # Process the response text
                self.memory.append({"role": "assistant", "content": response_text})
                print(block.content[0].text)
                break


            elif block.type == "function_call":

                logging.debug("Function call: %s, args: %s", block.name, block.arguments)
                args_dict = json.loads(block.arguments)  # <-- Parse JSON string to dict
                tool_result = self.act(block.name, args_dict)                                       
                self.memory.append({
                    "type": "function_call",
                    "call_id": block.call_id,
                    "name": block.name,
                    "arguments": block.arguments
                })
                # Now append the output
                self.memory.append({
                    "type": "function_call_output",
                    "call_id": block.call_id,
                    "output": str(tool_result)
                })
        api_response_2 = self.client.responses.create(
            model="gpt-4o",
            input=self.memory,
        )
        response_text = api_response_2.output_text
                    
        return response_text

    def act(self, name, args):
        logging.debug("Acting with tool: %s, args: %s", name, args)
        tool = self.tools.get(name)
        model_cls = self.tool_models.get(name)
        
        if tool is None or model_cls is None:

            logging.error(f"Tool '{name}' not found.")
            return f"Tool or model for '{name}' not found."

        try:
            params = model_cls(**args)
            result = tool(params)
            return result
     
        except Exception as e:
            logging.error(f"Error calling tool '{name}': {e}")
            return f"Error: {e}"