# Sales Agent - SDR AI Assistant

## Project Overview

This repository contains a Sales Development Representative (SDR) AI Assistant that helps sales teams analyze lead data, query leads based on various criteria, and draft personalized sales emails. The system uses OpenAI's language models to process natural language queries and generate appropriate responses.

## Repository Structure

```
Sales Agent/
├── analysis/                # Data analysis Jupyter notebooks
│   ├── 1. data_analysis.ipynb     # Initial data exploration and analysis
│   ├── 2. data_consolidation.ipynb # Data cleaning and consolidation
│   ├── 3. query.ipynb             # Query engine development
│   ├── 4. sdr.ipynb               # SDR functionality development
│   ├── 5. workflow.ipynb          # End-to-end workflow testing
│   └── pandasai.ipynb             # PandasAI implementation experiments
├── app/                    # Main application
│   ├── app.py                     # Streamlit application
│   ├── landing_page_leads.csv     # Processed leads data
│   ├── cleaned_leads.csv          # Cleaned leads dataset
│   └── saved_conversations/       # Directory for saved conversation logs
├── evaluation/             # Testing and evaluation scripts
│   ├── evals.ipynb               # Evaluation metrics and tests
│   ├── test_DraftEmail.ipynb     # Email drafting functionality tests
│   └── test_SDR.ipynb            # SDR functionality tests
├── .venv/                  # Virtual environment (not tracked)
└── openai_calls.json       # Log of API calls made to OpenAI
```

## Data Files

The system uses two primary data files:

1. **landing_page_leads.csv**: Contains the full dataset of sales leads with details such as lead number, source, origin, contact information, and conversion status.

2. **cleaned_leads.csv**: A processed version of the leads data with standardized formats, filled missing values, and optimized for querying.

Key columns in the datasets include:
- Lead Number (unique identifier)
- Lead Source (e.g., Google, Landing Page Submission, Organic Search)
- Lead Origin (e.g., API, Landing Page Submission)
- Converted (binary 0/1)
- Country
- City
- Email
- Company
- Lead Stage
- Lead Score

## Analysis Notebooks

### 1. data_analysis.ipynb
- Initial data exploration and visualization
- Analysis of data distribution, missing values, and column relationships
- Insights into lead sources, conversion rates, and geographic distribution

### 2. data_consolidation.ipynb
- Data cleaning and preprocessing
- Handling of missing values and duplicates
- Merging and standardizing data from multiple sources

### 3. query.ipynb
- Development of natural language query engine
- Implementation of pandas filtering mechanisms
- Testing and validation of query capabilities

### 4. sdr.ipynb
- Development of SDR assistant functionality
- Testing of natural language understanding for lead filtering
- Email drafting capability development

### 5. workflow.ipynb
- End-to-end workflow testing
- Integration of query, filtering, and email drafting capabilities
- Performance optimization

### pandasai.ipynb
- Experiments with PandasAI for natural language data querying
- Alternative approaches to handling complex queries
- Performance comparison with direct OpenAI API integration

## Application

The main application (`app/app.py`) is built using Streamlit and provides a user-friendly interface for:

1. **Natural Language Querying**: Users can ask questions in plain English to filter leads
2. **Lead Visualization**: Filtered leads are displayed in a data table
3. **Email Drafting**: The system can generate personalized sales emails for specific leads
4. **Conversation Management**: Chat interface for interacting with the SDR assistant
5. **Conversation Saving**: Ability to save conversations for future reference

Key components of the application:

- **SDR Assistant**: Processes natural language requests and orchestrates tool calls
- **Query Engine**: Converts natural language to pandas queries
- **Email Generator**: Creates personalized email drafts using lead data
- **Tool Call Handler**: Manages execution of various functions based on user requests

## Evaluation

The evaluation directory contains scripts for testing different components of the system:

### evals.ipynb
- Evaluates overall system performance
- Analyzes response quality and accuracy
- Compares different model configurations

### test_DraftEmail.ipynb
- Tests the email drafting functionality
- Verifies personalization features
- Validates email content for different lead types

### test_SDR.ipynb
- Comprehensive testing of the SDR assistant
- Query processing and response evaluation
- Tool calling sequence validation

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Sales-Agent.git
   cd Sales-Agent
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:
   ```
   # In .env file
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```
cd app
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Usage Examples

1. **Querying leads**:
   - "Show me leads from landing page"
   - "Find leads from Mumbai who haven't converted yet"
   - "Which leads came from Google with a high lead score?"

2. **Drafting emails**:
   - "Draft an email for lead 588124"
   - "Write a sales pitch for John Smith at lead number 654321 for our AI Solutions product"

## Development and Extension

The codebase is structured to facilitate easy extension and improvement:

- **Adding new tools**: Extend the `tools_primary` list in app.py
- **Improving queries**: Enhance the query generation system in `run_primary_llm_query()`
- **Customizing emails**: Modify the email templates in `generate_sales_email()`

## Notes

- The application uses OpenAI's API, which requires an API key and incurs usage costs
- The system is designed for demonstration purposes and may require optimization for production use
- Conversation logs are stored locally in the `saved_conversations` directory

## Future Improvements

- Enhanced lead scoring system
- Integration with CRM platforms
- Support for additional data sources
- Advanced email performance analytics
- Multi-user support with role-based access control

## License

[Your License Information Here]
