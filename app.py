import streamlit as st
import os
import pandas as pd
import json

# If you are using the same openai.py library as in your code:
from openai import OpenAI

# PandasAI + LLM
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI

# If using your custom local openai client:
# client = <YourClient>()
# Otherwise, if you want to do e.g.:
client = OpenAI()

# Set up a separate LLM object for PandasAI (for fallback):
llm_pandasai = PandasAI_OpenAI(model="gpt-4o-mini", temperature=0)

########################################
# Streamlit Page Config
########################################
st.set_page_config(page_title="SDR AI Assistant", layout="wide")
st.title(" Llama ")
st.caption("Your AI assistant for querying leads and drafting emails.")

########################################
# Session State Variables for UI Components
########################################
# Initialize dataframe and email display containers in session state
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "current_email" not in st.session_state:
    st.session_state.current_email = {"subject": "", "body": ""}

########################################
# Constants / Configuration
########################################
SDR_ASSISTANT_INSTRUCTIONS = """
You are a friendly and efficient assistant for a Sales Development Representative (SDR).
ALWAYS CHECK FOR INFORMATION IN THE CONVERSATION HISTORY BEFORE ASKING THE SDR FOR MORE INFORMATION. TOOLS MIGHT HAVE BEEN CALLED ALREADY.

Your job is to support the SDR in identifying high-quality leads and drafting effective, personalized sales emails.
You help the SDR save time by interpreting their natural language requests and converting them into meaningful actions using the tools available to you.

You have access to three tools:

1. nl_to_query
   - Use this tool to retrieve relevant leads from the database.
   - NEVER write pandas code yourself. Instead, rephrase the SDR’s request in natural language and call this tool.
   - Use this tool when the SDR asks to find or filter leads (e.g., "Show me leads from Mumbai who haven’t converted yet").
   - ALWAYS ASK FOR MISSING INFORMATION TO COMPLETE THE REQUEST.

2. draft_sales_email_by_lead_number
   - Use this tool to generate a personalized sales email for a specific lead.
   - You’ll need the lead number, product name, and the lead's first and last name.
   - The email will be drafted using data from the leads database to make it more relevant and engaging.
   - DO NOT call the nl_to_query tool for email drafting requests.
   - ALWAYS ASK FOR MISSING INFORMATION TO COMPLETE THE REQUEST if any details (e.g., lead number, first name, last name, or product) are missing.

3. send_email
   - Use this tool to send an email.
   - It takes in the subject, body, and recipient's email address, and returns a confirmation message.
   - After the SDR confirms that the drafted email is good, you must call this tool to "send" the email.

What you can help the SDR with:
- Finding specific leads (e.g., "Show me leads from Mumbai who haven’t converted yet")
- Filtering leads by stage, score, city, source, or conversion status
- Identifying hot leads based on attributes (e.g., score > 100, lead grade = A/B)
- Drafting follow-up or outreach emails customized to the lead
- Suggesting lead prioritization based on score, source, or engagement

Important Instructions:
- Always use nl_to_query to query the leads database. NEVER attempt to write the query directly.
- ALWAYS ASK FOR MISSING INFORMATION TO COMPLETE THE REQUEST.
- If the SDR wants to send a sales email, gather the lead number, the lead's first and last name, and the product before proceeding.
- DO NOT call nl_to_query for email drafting requests.
- If the SDR's request is vague, ask clarifying questions to ensure you understand their needs before proceeding.
- After the SDR confirms that the email draft is acceptable, collect the recipient's email address (if not provided) and call send_email with the subject, body, and email address.
- If you are unsure or need more clarification, politely ask the SDR for specifics before proceeding.

Stay helpful, relevant, and focused on helping the SDR close more leads efficiently.
"""

FILTER_COLUMNS = [
    "Lead Number", "Lead Source", "Lead Origin", "Do Not Email",
    "Do Not Call", "Converted", "Country", "Lead Stage", "City"
]

########################################
# Load Data (Cached)
########################################
@st.cache_data
def load_data():
    """
    Loads the CSVs for leads data.
    Adjust the file paths as per your system.
    """
    try:
        path_merged = "merged_leads.csv"
        path_cleaned = "cleaned_leads.csv"
        df_master = pd.read_csv(path_merged)
        df_cleaned_leads = pd.read_csv(path_cleaned)

        # If needed, rename columns so that 'Lead Number_x' is consistent:
        if 'Lead Number_x' not in df_master.columns and 'Lead Number' in df_master.columns:
            df_master = df_master.rename(columns={'Lead Number': 'Lead Number_x'})

        return df_master, df_cleaned_leads
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        st.stop()

df_master, df_cleaned_leads = load_data()
df_leads = df_master  # For email drafting


########################################
# Helper Functions
########################################
def generate_schema(dataframe: pd.DataFrame, columns: list) -> str:
    """
    Returns a JSON string describing the requested columns of a DataFrame:
    column name, data type, and up to five sample values.
    """
    schema_list = []
    if dataframe is None:
        return json.dumps({"error": "DataFrame not loaded"}, indent=2)

    for col in columns:
        if col in dataframe.columns:
            try:
                examples = dataframe[col].dropna().unique()[:5].tolist()
                schema_list.append({
                    "column_name": col,
                    "dtype": str(dataframe[col].dtype),
                    "examples": examples
                })
            except Exception as e:
                st.warning(f"Could not serialize examples for column '{col}': {e}")
                schema_list.append({
                    "column_name": col,
                    "dtype": str(dataframe[col].dtype),
                    "examples": ["Error fetching examples"]
                })

    return json.dumps(schema_list, indent=2)


def execute_pandas_query(dataframe: pd.DataFrame, query_str: str):
    """
    Evaluates a Python expression on a copy of the given DataFrame.
    Returns either a DataFrame, Series, or any resulting object.
    """
    if dataframe is None:
        st.error("Attempted to query a None DataFrame.")
        return None
    local_vars = {"df": dataframe.copy()}
    try:
        # Security note: eval() can be risky if query_str is user-provided.
        # Here it is only LLM-generated from known context, but be cautious in production.
        result = eval(query_str, {"pd": pd, "__builtins__": {}}, local_vars)
        return result
    except Exception as e:
        st.error(f"Error executing pandas query: {e}")
        return f"Error: {e}"


def parse_pandasai_output(output):
    import pandas as pd
    import matplotlib.pyplot as plt

    if isinstance(output, pd.DataFrame):
    
        print("→ Output is a DataFrame")
        st.session_state.current_dataframe = output.copy()
        return {"type": "dataframe", "data": output.to_dict(orient="records")}

    elif isinstance(output, (int, float, str, bool)):
        print(f"→ Output is a {type(output).__name__}: {output}")
        return {"type": "scalar", "data": output}

    elif isinstance(output, dict):
        print("→ Output is a dictionary")
        return {"type": "dict", "data": output}

    elif isinstance(output, (list, tuple)):
        print("→ Output is a list/tuple")
        return {"type": "list", "data": output}

    elif isinstance(output, plt.Figure):
        print("→ Output is a Matplotlib Figure (chart)")
        plt.show()
        return {"type": "figure", "data": output}

    else:
        print(f"→ Output is of unexpected type: {type(output)}")
        return {"type": "unknown", "data": output}
    


def nl_to_query_fallback(nl_query: str):
    """
    A fallback approach using PandasAI if the primary approach fails.
    This uses df_master for its broad columns, or you can switch to df_cleaned_leads.
    """
    try:
        sdf = SmartDataframe(df_master, config={
            "llm": llm_pandasai,
            "custom_prompts": {
                "pandas_code": (
                    f"You are analyzing a sales leads dataset. Columns include: "
                    f"{list(df_master.columns)}. Generate valid Pandas code."
                )
            },
            "verbose": False
        })
        response = sdf.chat(nl_query)
        parsed = parse_pandasai_output(response)
        return parsed
    except Exception as e:
        st.error(f"Fallback with PandasAI error: {e}")
        return {"type": "error", "data": f"Fallback error: {e}"}


def nl_to_query(user_request: str, dataframe: pd.DataFrame) -> dict | list | str | int | float:
    """
    Sends the natural language request to an LLM to generate a pandas query,
    executes it, and returns the result. Attempts multiple retries; if it fails,
    tries fallback with PandasAI.
    """
    if dataframe is None:
        return "Error: Data not loaded."

    schema_description = generate_schema(dataframe, FILTER_COLUMNS)
    max_retries = 1
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        prompt = f"""
            You are an expert data analyst.

            You have access to a tool called `execute_pandas_query` that accepts a Python expression as input and runs it on a DataFrame called `df`.

            Here is the DataFrame schema:
            {schema_description}

            Below are a few one-shot examples of valid filtering queries:

            Example 1:
            User Request: "List all leads from City 'Metropolis' that have not been converted"
            Expected tool call: execute_pandas_query("df[(df['City'] == 'Metropolis') & (df['Converted'] == 0)]")

            Example 2:
            User Request: "Show me leads from Source 'Web' with a lead score above 100"
            Expected tool call: execute_pandas_query("df[(df['Lead Source'] == 'Web') & (df['Lead Score'] > 100)]")

            Example 3:
            User Request: "Find leads that are qualified and in the age group '30-35'"
            Expected tool call: execute_pandas_query("df[(df['Lead Stage'] == 'Qualified') & (df['Age'] == '30-35')]")


            Now, a user has asked to filter the DataFrame with this request:
            '{user_request}'

            You should call the tool with the appropriate pandas filtering expression using `df`.
            Understand the schema and the valid values in the columns.
            rephrase the user query with available values within the particular column.
            Strictly align with the query and do not give any other information or irrelevant details.
            Only call the tool. Do not answer the query yourself.

        """

        # We define a mini tool set for generating the query:
        tools_secondary = [
            {
                "type": "function",
                "function": {
                    "name": "execute_pandas_query",
                    "description": "Run a pandas query string on a DataFrame called df.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_str": {
                                "type": "string",
                                "description": "The Python code string to evaluate on df."
                            }
                        },
                        "required": ["query_str"]
                    }
                }
            }
        ]

        # Call the LLM to produce the query (and hopefully call the function)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a pandas expert."},
                    {"role": "user", "content": prompt}
                ],
                tool_choice="auto",
                tools=tools_secondary,
                temperature=0
            )
        except Exception as e:
            #if primary call fails we fallback to pandasai
            st.error(f"Error calling LLM for query generation: {e}")
            if attempt == max_retries:
                return nl_to_query_fallback(user_request)  # final fallback
            continue

        # Check if the LLM tried to call `execute_pandas_query`
        llm_msg = response.choices[0].message
        if llm_msg.tool_calls:
            # Take the first tool call
            tool_call = llm_msg.tool_calls[0]
            if tool_call.function.name == "execute_pandas_query":
                query_str = json.loads(tool_call.function.arguments)["query_str"]
                tool_result = execute_pandas_query(dataframe, query_str)

                if isinstance(tool_result, str) and tool_result.startswith("Error:"):
                    # Query execution error
                    st.warning(f"LLM query attempt {attempt} failed: {tool_result}")
                    #if tool call fails we fallback to pandasai
                    if attempt == max_retries:
                        fallback_res = nl_to_query_fallback(user_request)
                        return fallback_res.get("data", fallback_res)
                    continue
                else:
                    # If success, return the data in a standard format
                    if isinstance(tool_result, pd.DataFrame):
                        # Store the dataframe in session state for display
                        st.session_state.current_dataframe = tool_result.copy()
                        # limit rows for large data
                        limit = 50
                        if len(tool_result) > limit:
                            return tool_result.head(limit).to_dict(orient="records")
                        else:
                            return tool_result.to_dict(orient="records")
                    elif isinstance(tool_result, pd.Series):
                        return tool_result.tolist()
                    else:
                        return tool_result
            else:
                # Unexpected tool call
                st.warning(f"Unexpected tool called: {tool_call.function.name}")
                if attempt == max_retries:
                    return nl_to_query_fallback(user_request)
                continue
        else:
            # No tool call from the LLM
            direct_answer = llm_msg.content
            st.warning("LLM returned text without a tool call. Will attempt fallback or retry.")
            if attempt == max_retries:
                fallback = nl_to_query_fallback(user_request)
                return fallback.get("data", fallback)
            continue

    # If we exit the loop, we do a final fallback
    final_fallback = nl_to_query_fallback(user_request)
    return final_fallback.get("data", final_fallback)


def draft_sales_email_by_lead_number(
    lead_number: int,
    product: str,
    first_name: str,
    last_name: str,
    dataframe_leads: pd.DataFrame = None
) -> str:
    """
    Drafts a personalized sales email for a given lead number using data from the provided DataFrame.
    """
    st.info(f"Drafting email for lead: {lead_number} | product: {product} | {first_name} {last_name}")
    if dataframe_leads is None:
        dataframe_leads = df_leads

    agent_name = "Your AI SDR Assistant"
    agent_company = "Your Company"
    agent_contact_information = "you@example.com"

    if "Lead Number_x" not in dataframe_leads.columns:
        st.error("Column 'Lead Number_x' not found in the leads data. Possibly rename or adjust.")
        return "Error: 'Lead Number_x' column not found. Cannot draft email."

    # Filter by lead number
    lead_data = dataframe_leads[dataframe_leads["Lead Number_x"] == int(lead_number)]
    if lead_data.empty:
        return f"Error: Lead with number {lead_number} not found."

    lead = lead_data.iloc[0].to_dict()
    # subject & body
    email_subject = f"Discover How {product} Can Benefit {first_name} {last_name}"
    email_body = (
        f"Hi {first_name},\n\n"
        f"I hope you're doing well. My name is {agent_name} from {agent_company}.\n\n"
        f"I'm reaching out because I noticed "
        f"{lead.get('Company', 'your company')} "
        f"{'based in ' + lead['City'] + ', ' if lead.get('City') else ''}"
        f"might be a great fit for our {product}. "
        "We specialize in helping businesses like yours "
        f"{'in the ' + lead['Industry'] + ' sector ' if lead.get('Industry') else ''}"
        "improve efficiency and drive growth.\n\n"
        f"I'd love to schedule a brief call to discuss how {product} could benefit you. "
        "Let me know if you have availability next week!\n\n"
        "Best regards,\n"
        f"{agent_name}\n"
        f"{agent_company}\n"
        f"{agent_contact_information}"
    )
    email_content = f"Subject: {email_subject}\n---\n{email_body}"
    
    # Store the email in session state for display
    st.session_state.current_email = {
        "subject": email_subject,
        "body": email_body,
        "recipient": f"{first_name} {last_name}"
    }
    
    return email_content


def send_email(subject: str, body: str, email_id: str) -> str:
    """
    Simulates sending an email. In a real app, you'd integrate with your email provider.
    """
    confirmation = f"Success! (Simulation) Email with subject '{subject}' was sent to {email_id}."
    st.success(confirmation)
    return confirmation


########################################
# Tools (for the Main Orchestrator)
########################################
tools_primary = [
    {
        "type": "function",
        "function": {
            "name": "nl_to_query",
            "description": (
                "Convert a natural language query into a valid pandas filtering expression to retrieve lead data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nl_query": {
                        "type": "string",
                        "description": "The user's natural language filter request."
                    }
                },
                "required": ["nl_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "draft_sales_email_by_lead_number",
            "description": (
                "Drafts a personalized sales email for a given lead number by looking up the lead data from the DataFrame."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_number": {
                        "type": "integer",
                        "description": "Unique integer identifier for the lead."
                    },
                    "product": {
                        "type": "string",
                        "description": "The product or service to pitch."
                    },
                    "first_name": {
                        "type": "string",
                        "description": "First name of the lead/contact person."
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Last name of the lead/contact person."
                    }
                },
                "required": ["lead_number", "product", "first_name", "last_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Sends an email with the provided subject, body, and recipient email address, then returns a confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject of the email."
                    },
                    "body": {
                        "type": "string",
                        "description": "The content of the email body."
                    },
                    "email_id": {
                        "type": "string",
                        "description": "The recipient's email address."
                    }
                },
                "required": ["subject", "body", "email_id"]
            }
        }
    }
]

########################################
# Tool Call Handler
########################################
def handle_tool_call(tool_call):
    """
    Executes the tool function based on the 'function.name' and the JSON arguments.
    """
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except Exception as e:
        return f"Error decoding tool arguments: {e}"

    if function_name == "nl_to_query":
        nl_query = arguments["nl_query"]
        return nl_to_query(nl_query, df_cleaned_leads)

    elif function_name == "draft_sales_email_by_lead_number":
        lead_number = arguments["lead_number"]
        product = arguments["product"]
        first_name = arguments["first_name"]
        last_name = arguments["last_name"]
        return draft_sales_email_by_lead_number(
            lead_number, product, first_name, last_name, df_leads
        )

    elif function_name == "send_email":
        subject = arguments["subject"]
        body = arguments["body"]
        email_id = arguments["email_id"]
        return send_email(subject, body, email_id)

    else:
        return f"Error: Unknown tool call '{function_name}'."


########################################
# Main Orchestrator (Similar to run_sdr_agent)
########################################
def run_sdr_agent(user_message: str, session_memory: dict) -> str:
    """
    1. Append user message to memory
    2. Call LLM with the entire conversation + tools
    3. If tool is called, handle it, add the result to memory, re-call LLM
    4. Return final text
    """
    session_memory["sdr"]["messages"].append({"role": "user", "content": user_message})

    try:
        # Step 1: main call
        response = client.chat.completions.create(
            messages=session_memory["sdr"]["messages"],
            model="o3-mini",  # or "gpt-4o-mini"
            tool_choice="auto",
            tools=tools_primary,
            # temperature=0,
        )
        response_msg = response.choices[0].message
        # Save assistant's response or tool-calling attempt in memory:
        session_memory["sdr"]["messages"].append(response_msg)

        # Step 2: check for tool calls
        if response_msg.tool_calls:
            all_tool_results = []
            for tc in response_msg.tool_calls:
                tool_result = handle_tool_call(tc)
                all_tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                })

            # Add the tool results to memory
            session_memory["sdr"]["messages"].extend(all_tool_results)

            # Step 3: follow-up call with the tool results in memory
            followup = client.chat.completions.create(
                messages=session_memory["sdr"]["messages"],
                model="gpt-4o"
            )
            final_msg = followup.choices[0].message.content
            session_memory["sdr"]["messages"].append({"role": "assistant", "content": final_msg})
            return final_msg
        else:
            # No tool call, just return text
            final_text = response_msg.content
            return final_text

    except Exception as e:
        st.error(f"Error in run_sdr_agent: {e}")
        return f"Error: {e}"


########################################
# Streamlit UI Layout
########################################
# Create a 2-column layout for main chat and side panels
col1, col2 = st.columns([3, 2])

# Left column for chat interface
with col1:
    st.header("Chat with your SDR Assistant")
    
    # Initialize if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = {
            "sdr": {
                "messages": [{"role": "system", "content": SDR_ASSISTANT_INSTRUCTIONS}]
            }
        }
        # Provide an initial AI greeting
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hi! How can I help you with your leads today?"}
        )

    # Show the conversation so far
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Accept user input from chat
    user_input = st.chat_input("Ask your SDR assistant...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            assistant_answer = run_sdr_agent(user_input, st.session_state.memory)

        # Display final assistant response
        st.session_state.messages.append({"role": "assistant", "content": assistant_answer})
        with st.chat_message("assistant"):
            st.markdown(assistant_answer)
        
        # Auto-scroll chat to bottom (requires JavaScript - not perfect but helps)
        js = '''
        <script>
            function scroll_to_bottom() {
                var chatbox = document.querySelector('.stChatContainer');
                if (chatbox) chatbox.scrollTop = chatbox.scrollHeight;
            }
            window.addEventListener('load', scroll_to_bottom);
        </script>
        '''
        st.components.v1.html(js)

# Right column for dataframe and email display
with col2:
    # Dataframe display box
    with st.expander("Filtered Leads", expanded=True):
        st.subheader("Lead Search Results")
        if st.session_state.current_dataframe is not None and not st.session_state.current_dataframe.empty:
            st.dataframe(st.session_state.current_dataframe)
        else:
            st.info("No lead data to display yet. Ask me to find leads!")
    
    # Email display box
    with st.expander("Email Draft", expanded=True):
        st.subheader("Current Email Draft")
        if st.session_state.current_email["subject"]:
            st.markdown(f"**To:** {st.session_state.current_email.get('recipient', 'Recipient')}")
            st.markdown(f"**Subject:** {st.session_state.current_email['subject']}")
            st.text_area("Email Body", value=st.session_state.current_email["body"], height=200, key="email_body_display", disabled=True)
            
            # Add send button for convenience
            recipient_email = st.text_input("Recipient Email Address")
            if st.button("Send Email") and recipient_email:
                confirmation = send_email(
                    st.session_state.current_email["subject"],
                    st.session_state.current_email["body"],
                    recipient_email
                )
                st.success(confirmation)
        else:
            st.info("No email draft yet. Ask me to create an email for a lead!")
