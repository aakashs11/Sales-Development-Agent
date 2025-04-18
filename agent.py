class agent:
    def __init__(self, name, client, developer_instructions, tools):
        self.name = name
        self.memory = [{"role": "developer", "content": developer_instructions}]
        self.tools = tools if tools is not None else []
        self.tools = []

    def response(self, input_data):
        # Generate a response based on input data
        self.memory.append({"role": "user", "content": input_data})
        api_response_1= self.client.responses.create(
            model="gpt-4o",
            messages=[{"role": "developer", "content": input_data}],
        )

        # Process the API response
        for block in api_response_1.output:

            if block.type == "message":

                response_text = api_response_1.output_text
                # Process the response text
                self.memory.append({"role": "assistant", "content": response_text})
                print(block.content[0].text)

            elif block.type == "tool_calls":

                for call in block.calls:

                    print(call.name)
                    print(call.args)
                    tool_response = self.act(call.name, call.args)
                    self.memory.append({"role": "tool", "content": tool_response})
                    api_response_2 = self.client.responses.create(
                        model="gpt-4o",
                        messages=self.memory,
                    )
                    response_text = api_response_2.output_text
                    
        return response_text

    def act(self, name, args):
        # Perform an action based on name and args
        if name == "call_database":
            action = f"Calling database with args: {args}"

        return f"{self.name} performs action: {action}"