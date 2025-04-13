# Sales Development Agent

## Overview
The Sales Development Agent is a Streamlit-based AI-powered assistant designed to support Sales Development Representatives (SDRs) in managing leads efficiently. It can help filter leads, draft personalized sales emails, and send them, all with a few clicks.

## Features
- **Lead Management:** 
  - Filter leads by attributes such as city, stage, score, and more.
  - Query leads using natural language commands.
- **Email Drafting:** 
  - Generate personalized sales emails based on lead data.
  - Customize email drafts before sending.
- **Email Sending:** 
  - Send emails directly with subject, body, and recipient's address.
- **Data Insights:**
  - Load and analyze lead data from CSV files.

## Technologies Used
- **Python**
- **Streamlit:** For building the web interface.
- **OpenAI API:** For natural language processing and lead querying.
- **Pandas:** For data manipulation and analysis.
- **PandasAI:** For intelligent DataFrame operations.
- **Matplotlib:** For visualizations.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/aakashs11/Sales-Development-Agent.git
   cd Sales-Development-Agent
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the OpenAI API key:
   - Add your OpenAI API key to the environment variables or `st.secrets` in Streamlit.
4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Usage
1. **Start the App:** Open the Streamlit application using the command above. It will launch in your default web browser.
2. **Chat Interface:**
   - Use the chat input field to interact with the AI assistant.
   - Example queries:
     - "Show me leads from Mumbai who haven’t converted yet."
     - "Create a sales email for lead number 123 with product 'Tech Solutions'."
3. **Filtered Leads:**
   - View filtered lead data in the 'Filtered Leads' section.
   - Analyze data to prioritize and target specific leads.
4. **Email Drafting and Sending:**
   - Draft personalized emails using lead data.
   - Edit and send emails directly from the app.
5. **Save Conversations:**
   - Save chat interactions and generated data for future reference.

## Project Structure
```
Sales-Development-Agent/
│
├── app/
│   └── app.py                 # Main application file
├── landing_page_leads.csv     # Example dataset
├── cleaned_leads.csv          # Example cleaned dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for any feature requests or bugs.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments
Special thanks to the OpenAI and Streamlit communities for their contributions to AI and web development.
