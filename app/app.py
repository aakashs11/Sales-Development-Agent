import streamlit as st
st.set_page_config(page_title="Debugging App", layout="wide")
st.info("🟢 App is starting...")
import json
import os

import pandas as pd
import streamlit as st
from openai import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasAI_OpenAI
import os
# Robust API key getter: handles both local and Streamlit Cloud
def get_api_key():
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

api_key = get_api_key()

if not api_key:
    st.error("OpenAI API key not found in environment or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
llm_pandasai = PandasAI_OpenAI(api_token=api_key, model="gpt-4o", temperature=0)
import matplotlib.pyplot as plt
import pandas as pd

########################################
# Streamlit Page Config
########################################
st.set_page_config(page_title="Sales Development Representative Agent", layout="wide")
st.title(" SDR AI Assistant")
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

1. run_primary_llm_query
   - Use this tool to retrieve relevant leads from the database.
   - NEVER write pandas code yourself. Instead, rephrase the SDR’s request in natural language and call this tool.
   - Use this tool when the SDR asks to find or filter leads (e.g., "Show me leads from Mumbai who haven’t converted yet").
   - Give names of the leads when the SDR asks for a list of leads along with their details.

2. generate_sales_email
   - Use this tool to generate a personalized sales email for a specific lead.
   - You’ll need the lead number, product name, and the lead's first and last name.
   - The email will be drafted using data from the leads database to make it more relevant and engaging.
   - Check the conversation history for the lead number, product, and names and if the information is not available, ask the run_primary_llm_query tool to get the information.
   - DO NOT call the run_primary_llm_query tool for email drafting requests unless information is missing.
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
- Always use run_primary_llm_query to query the leads database. NEVER attempt to write the query directly.
- ALWAYS ASK FOR MISSING INFORMATION TO COMPLETE THE REQUEST.
- If the SDR wants to send a sales email, gather the lead number, the lead's first and last name, and the product before proceeding.
- DO NOT call run_primary_llm_query for email drafting requests.
- If the SDR's request is vague, ask clarifying questions to ensure you understand their needs before proceeding.
- After the SDR confirms that the email draft is acceptable, collect the recipient's email address (if not provided) and call send_email with the subject, body, and email address.
- If you are unsure or need more clarification, politely ask the SDR for specifics before proceeding.

Stay helpful, relevant, and focused on helping the SDR close more leads efficiently.
"""

FILTER_COLUMNS = [
    "Lead Number",
    "Lead Source",
    "Lead Origin",
    "Do Not Email",
    "Do Not Call",
    "Converted",
    "Country",
    "Lead Stage",
    "City",
    "Email",
    "Company",
]


########################################
# Load Data (Cached)
########################################
@st.cache_data
def load_leads_dataset():
    """
    Loads the CSVs for leads data.
    Adjust the file paths as per your system.
    """
    try:
        #filtered leads and cleaned leads are in the same directory as this script
        # Adjust the path as necessary
        path_merged = "landing_page_leads.csv"
        path_cleaned = "cleaned_leads.csv"
        df_master = pd.read_csv(path_merged)
        df_cleaned_leads = pd.read_csv(path_cleaned)
        print(df_cleaned_leads.head())
        print(df_master.head())
        # If needed, rename columns so that 'Lead Number_x' is consistent:
        if (
            "Lead Number_x" not in df_master.columns
            and "Lead Number" in df_master.columns
        ):
            df_master = df_master.rename(columns={"Lead Number": "Lead Number_x"})

        return df_master, df_cleaned_leads
    except FileNotFoundError as e:
        # st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        # st.error(f"Unexpected error loading data: {e}")
        st.stop()


df_master, df_cleaned_leads = load_leads_dataset()
df_leads = df_master  # For email drafting


########################################
# Helper Functions
########################################
def describe_dataframe_columns(dataframe: pd.DataFrame, columns: list) -> str:
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
                schema_list.append(
                    {
                        "column_name": col,
                        "dtype": str(dataframe[col].dtype),
                        "examples": examples,
                    }
                )
            except Exception as e:
                st.warning(f"Could not serialize examples for column '{col}': {e}")
                schema_list.append(
                    {
                        "column_name": col,
                        "dtype": str(dataframe[col].dtype),
                        "examples": ["Error fetching examples"],
                    }
                )

    return json.dumps(schema_list, indent=2)


def run_query_on_leads(dataframe: pd.DataFrame, query_str: str):
    """
    Evaluates a Python expression on a copy of the given DataFrame.
    Returns either a DataFrame, Series, or any resulting object.
    """
    print("Entering run_query_on_leads")
    if dataframe is None:
        st.error("Attempted to query a None DataFrame.")
        return None
    local_vars = {"df": dataframe.copy()}
    try:
        print(f"Executing query: {query_str}")
        result = eval(query_str, {"__builtins__": {}}, local_vars)
        print(f"Query executed successfully: {result}")
        return result
    except Exception as e:
        st.error(f"Error executing pandas query: {e}")
        return f"Error: {e}"


def normalize_llm_output(output):

    if isinstance(output, pd.DataFrame):

        print("→ Output is a DataFrame")

        print(f"→ DataFrame shape: {output.shape}")
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


def run_secondary_llm_query(nl_query: str):
    """
    A fallback approach using PandasAI if the primary approach fails.
    This uses df_master for its broad columns, or you can switch to df_cleaned_leads.
    Retries two times if there's an error or the DataFrame is empty.
    """
    print(f"Fallback NL to Query: {nl_query}")

    columns = df_master.columns.tolist()
    sdf = SmartDataframe(
        df_master,
        config={
            "llm": llm_pandasai,
            "custom_prompts": {
                "pandas_code": f" You are analyzing a sales leads dataset. Columns include {columns} Be careful with column names and types.",
            },
            "verbose": True,
        },
    )
    try:
        response = sdf.chat(nl_query)
        st.session_state.current_dataframe = response
        return normalize_llm_output(response)
    except Exception as e:
        return f"Error: {e}"


def run_primary_llm_query(user_request: str) -> dict | list | str | int | float:
    """
    Sends the natural language request to an LLM to generate a pandas query,
    executes it, and returns the result. Attempts multiple retries; if it fails,
    tries fallback with PandasAI.
    """
    print(f"NL to Query: {user_request}")
    dataframe = df_cleaned_leads

    schema_description = describe_dataframe_columns(dataframe, FILTER_COLUMNS)
    max_retries = 1
    attempt = 0

    while attempt < max_retries:

        prompt = f"""
            You are an expert data analyst.

            You have access to a tool called `run_query_on_leads` that accepts a Python expression as input and runs it on a DataFrame called `df`.

            Here is the DataFrame schema:
            {schema_description}

            Below are a few one-shot examples of valid filtering queries:

            Example 1:
            User Request: "List all leads from City 'Metropolis' that have not been converted"
            Expected tool call: run_query_on_leads("df[(df['City'] == 'Metropolis') & (df['Converted'] == 0)]")

            Example 2:
            User Request: "Show me leads from Source 'Web' with a lead score above 100"
            Expected tool call: run_query_on_leads("df[(df['Lead Source'] == 'Web') & (df['Lead Score'] > 100)]")

            Example 3:
            User Request: "Find leads that are qualified and in the age group '30-35'"
            Expected tool call: run_query_on_leads("df[(df['Lead Stage'] == 'Qualified') & (df['Age'] == '30-35')]")


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
                    "name": "run_query_on_leads",
                    "description": "Run a pandas query string on a DataFrame. Use 'df' as the variable name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_str": {
                                "type": "string",
                                "description": "Python code string to evaluate on the DataFrame. Use 'df' as the variable name.",
                            }
                        },
                        "required": ["query_str"],
                    },
                },
            },
        ]
        messages = [
            {"role": "system", "content": "You are a pandas expert."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            stream=False,
            tool_choice="auto",
            tools=tools_secondary,
            temperature=0,
        )
        # 1) Try calling GPT‐4o
        print("Calling LLM for query generation...\n\n")
        try:
            response_content = response.choices[0].message.content
            if response_content:
                print(response_content.strip())
            else:
                print("No direct LLM response. Possibly a tool call only.")

        except Exception as e:
            print(f"Error extracting LLM response: {e}")
            st.warning(f"Error extracting LLM response: {e}")
            continue

        # 2) Check if a tool call was generated
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Tool call from Query Function: {tool_call}")
        except Exception as e:
            print("Failed to extract a tool call from the LLM response:", e)
            attempt += 1
            continue  # Retry the LLM call

        # Validate query by attempting to execute it
        try:
            tool_result = handle_tool_call(tool_call)
            print("tool result from Query Function:", tool_result)
            if tool_result == [] or None:
                print("Tool call returned empty or invalid result. Retrying...")
                attempt += 1
                return run_secondary_llm_query(user_request)
            return tool_result  # Return if successful
        except Exception as e:
            print(f"Query execution failed on attempt {attempt+1}: {e}")
            attempt += 1
            print("Retrying query generation...\n")
            # If the LLM response is empty or invalid, retry with pandas query generation
            return run_secondary_llm_query(user_request)


def generate_sales_email(
    lead_number: str,
    product: str,
    first_name: str,
    last_name: str,
    dataframe_leads: pd.DataFrame = None,
) -> str:
    """
    Drafts a personalized sales email for a given lead number using data from the provided DataFrame.
    """
    st.info(
        f"Drafting email for lead: {lead_number} | product: {product} | {first_name} {last_name}"
    )
    if dataframe_leads is None:
        dataframe_leads = df_master

    print("entered email drafting function")
    agent_name = "Jane Doe"  # Placeholder for the agent's name
    agent_company = "Tech Solutions"  # Placeholder for the agent's company
    agent_contact_information = "jane.doe@xyz.com"


    email_body = {""}

    if "Lead Number_x" not in dataframe_leads.columns:
        # st.error("Column 'Lead Number_x' not found in the leads data. Possibly rename or adjust.")
        return "Error: 'Lead Number_x' column not found. Cannot draft email."

    # Filter by lead number
    lead_data = dataframe_leads[dataframe_leads["Lead Number_x"] == int(lead_number)]
    if lead_data.empty:
        return f"Error: Lead with number {lead_number} not found."

    lead = lead_data.iloc[0].to_dict()
    # subject & body
    email_subject = f"Discover How {product} Can Benefit {first_name, last_name}"
    email_body = (
        f"Hi {first_name} {last_name},\n\n"
        f"I hope you're doing well. My name is {agent_name} and I work as a Sales Engineer at {agent_company}.\n\n"
        f"I'm reaching out because I believe our {product} can significantly help {lead.get('Company', 'ABC')} meet its goals. "
        "We have a strong track record of helping organizations in your industry optimize their processes and drive success.\n\n"
        "I'd love to arrange a brief call to discuss how we can support your goals and explore further benefits of our solution.\n\n"
        "Looking forward to hearing from you.\n\n"
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
        "recipient": f"{first_name} {last_name}",
    }

    return email_content


def send_email(subject: str, body: str, email_id: str) -> str:
    """
    Simulates sending an email. In a real app, you'd integrate with your email provider.
    """
    confirmation_message = (
        f"Dummy email sent to {email_id} with subject '{subject} and Body {body}'."
    )
    st.success(confirmation_message)
    return confirmation_message


########################################
# Tools (for the Main Orchestrator)
########################################
tools_primary = [
    {
        "type": "function",
        "function": {
            "name": "run_primary_llm_query",
            "description": "Convert a natural language query into a valid pandas filtering expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nl_query": {
                        "type": "string",
                        "description": "The user's natural language filter request (e.g., 'show leads from Delhi with conversion 0').",
                    }
                },
                "required": ["nl_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sales_email",
            "description": (
                "Drafts a personalized sales email for a given lead number by looking up the lead data "
                "from the DataFrame and injecting a product pitch and lead's name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_number": {
                        "type": "integer",
                        "description": "Unique integer identifier for the lead.",
                    },
                    "product": {
                        "type": "string",
                        "description": "The product or service to pitch.",
                    },
                    "first_name": {
                        "type": "string",
                        "description": "First name of the lead/contact person.",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Last name of the lead/contact person.",
                    },
                },
                "required": ["lead_number", "product", "first_name", "last_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Sends an email with the provided subject, body, and recipient email address, "
                "then returns a confirmation message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject of the email.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The content of the email body.",
                    },
                    "email_id": {
                        "type": "string",
                        "description": "The recipient's email address.",
                    },
                },
                "required": ["subject", "body", "email_id"],
            },
        },
    },
]


########################################
# Tool Call Handler
########################################
def handle_tool_call(tool_call):
    """
    Handles the tool call by extracting the function name and arguments,
    running the appropriate function, and returning the result.
    """
    dataframe = df_cleaned_leads
    print("Tool call received at Handler:")
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "run_query_on_leads":
        print("Executing function:", function_name)
        query_str = arguments["query_str"]
        result = run_query_on_leads(dataframe, query_str)
        # add result to session state for display
        st.session_state.current_dataframe = result
        #send only lead number, lead stage, city and company name to the assistant to save tokens
        result = result[["Lead Number", "Lead Stage", "City", "Company","Lead Source", "Email"]]

        if isinstance(result, pd.DataFrame):
            return result.to_dict(orient="records")
        elif isinstance(result, pd.Series):
            return result.tolist()
        else:
            return result
    elif function_name == "run_primary_llm_query":
        print("Converting natural language to query:", arguments["nl_query"])
        nl_query = arguments["nl_query"]
        result = run_primary_llm_query(nl_query)
        return result
    elif function_name == "generate_sales_email":
        print("Drafting sales email by lead number:", arguments["lead_number"])

        lead_number = arguments["lead_number"]
        product = arguments["product"]
        first_name = arguments["first_name"]
        last_name = arguments["last_name"]
        result = generate_sales_email(
            lead_number, product, first_name, last_name
        )
        return result
    elif function_name == "send_email":
        print("Sending email:", arguments)
        email_subject = arguments["subject"]
        email_body = arguments["body"]
        email_id = arguments["email_id"]
        result = send_email(email_subject, email_body, email_id)
        return result


########################################
# Main Orchestrator (Similar to handle_sdr_conversation)
########################################
def handle_sdr_conversation(user_message: str, session_memory: dict) -> str:
    """
    1. Append user message to memory
    2. Call LLM with the entire conversation + tools
    3. If tool is called, handle it, add the result to memory, re-call LLM
    4. Return final text
    """
    print(f"Running SDR agent with message: {user_message}")
    session_memory["sdr"]["messages"].append({"role": "user", "content": user_message})

    try:
        # Step 1: main call
        print("Calling LLM with messages:")
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
        print("Response message:", response_msg)
        if response_msg.tool_calls:
            print("Tool calls detected.")
            all_tool_results = []
            for tc in response_msg.tool_calls:
                tool_result = handle_tool_call(tc)
                all_tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": (
                            json.dumps(tool_result)[:16000]  # Limit to 16000 chars
                            if not isinstance(tool_result, str)
                            else tool_result[:16000]  # Limit to 16000 chars
                        ),
                    }
                )

            # Add the tool results to memory
            session_memory["sdr"]["messages"].extend(all_tool_results)

            # Step 3: follow-up call with the tool results in memory
            followup = client.chat.completions.create(
                messages=session_memory["sdr"]["messages"], model="o3-mini",  # or "gpt-4o-mini"
            )
            final_msg = followup.choices[0].message.content
            session_memory["sdr"]["messages"].append(
                {"role": "assistant", "content": final_msg}
            )
            return final_msg
        else:
            # No tool call, just return text
            final_text = response_msg.content
            return final_text

    except Exception as e:
        # st.error(f"Error in handle_sdr_conversation: {e}")
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
            {
                "role": "assistant",
                "content": "Hi! How can I help you with your leads today?",
            }
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
            assistant_answer = handle_sdr_conversation(user_input, st.session_state.memory)

        # Display final assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_answer}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_answer)
        # ----------------------------
    # SAVE CONVERSATION BUTTON
    # ----------------------------
    if st.button("Save Conversation"):
        import time
        # Convert all message objects to serializable dictionaries
        def serialize_message(message):
            if hasattr(message, 'model_dump'):  # For OpenAI SDK objects
                return message.model_dump()
            elif isinstance(message, dict):
                return message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
            
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_conversations")
        os.makedirs(save_dir, exist_ok=True)
        # Create a timestamped filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"{timestamp_str}_conversation.json")
        
        raw_log = st.session_state.memory["sdr"]["messages"]
        full_log = [serialize_message(msg) for msg in raw_log]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(full_log, f, ensure_ascii=False, indent=2)

        st.success(f"Conversation saved to {filename}")

        # # Auto-scroll chat to bottom (requires JavaScript - not perfect but helps)
        # js = '''
        # <script>
        #     function scroll_to_bottom() {
        #         var chatbox = document.querySelector('.stChatContainer');
        #         if (chatbox) chatbox.scrollTop = chatbox.scrollHeight;
        #     }
        #     window.addEventListener('load', scroll_to_bottom);
        # </script>
        # '''
        # st.components.v1.html(js)

# Right column for dataframe and email display
with col2:
    # Dataframe display box
    with st.expander("Filtered Leads", expanded=True):
        st.subheader("Lead Search Results")
        if (
            st.session_state.current_dataframe is not None
            and not st.session_state.current_dataframe.empty
        ):
            st.dataframe(st.session_state.current_dataframe)
        else:
            st.info("No lead data to display yet. Ask me to find leads!")

    # Email display box
    with st.expander("Email Draft", expanded=True):
        st.subheader("Current Email Draft")
        if st.session_state.current_email["subject"]:
            st.markdown(
                f"**To:** {st.session_state.current_email.get('recipient', 'Recipient')}"
            )
            st.markdown(f"**Subject:** {st.session_state.current_email['subject']}")
            st.text_area(
                "Email Body",
                value=st.session_state.current_email["body"],
                height=200,
                key="email_body_display",
                disabled=True,
            )

            # Add send button for convenience
            recipient_email = st.text_input("Recipient Email Address")
            if st.button("Send Email") and recipient_email:
                confirmation = send_email(
                    st.session_state.current_email["subject"],
                    st.session_state.current_email["body"],
                    recipient_email,
                )
                st.success(confirmation)
        else:
            st.info("No email draft yet. Ask me to create an email for a lead!")
