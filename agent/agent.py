import logging
import json
class Agent:
    def __init__(self, name, client, developer_instructions, tool_schemas=None, tool_models=None, tools=None, memory=None):
        self.name = name
        self.client = client
        if memory and len(memory) > 0:
            self.memory = memory
        else:
            self.memory = [{"role": "developer", "content": developer_instructions}]

        self.tools = tools if tools is not None else {}  
        self.tool_schemas = tool_schemas if tool_schemas is not None else []
        self.tool_models = tool_models if tool_models is not None else {}

    def response(self, input_data):
        # Generate a response based on input data
        self.memory.append({"role": "user", "content": input_data})
        tool_output = None
        thoughts = []
        actions = []
        
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
                response_text = api_response_1.output_text
                self.memory.append({"role": "assistant", "content": response_text})
                thoughts.append("Generated response without tool call.")
                break


            elif block.type == "function_call":
                args_dict = json.loads(block.arguments)
                tool_result = self.act(block.name, args_dict)
                tool_output = tool_result
                actions.append({
                    "type": "function_call",
                    "tool": block.name,
                    "arguments": args_dict
                })
                self.memory.append({
                    "type": "function_call",
                    "call_id": block.call_id,
                    "name": block.name,
                    "arguments": block.arguments
                })
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
                    
        return {
            "memory": self.memory,
            "response": response_text,
            "thoughts": thoughts,
            "actions": actions,
            "tool_output": tool_output,
            "state": {
                "current_agent": self.name
            },
            "metadata": {},
            "error": None,
        }

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