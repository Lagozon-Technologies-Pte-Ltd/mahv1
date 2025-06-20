from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
import plotly.graph_objects as go, plotly.express as px
import openai, yaml, os, csv,pandas as pd, base64, uuid
from configure import gauge_config
from pydantic import BaseModel
from io import BytesIO, StringIO
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import Field
from langchain_openai import ChatOpenAI
from newlangchain_utils import *
from dotenv import load_dotenv
from state import session_state, session_lock
from typing import Optional
from starlette.middleware.sessions import SessionMiddleware  # Correct import
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from starlette.middleware.base import BaseHTTPMiddleware


import automotive_wordcloud_analysis as awa
import zipfile
from wordcloud import WordCloud
from table_details import get_table_details, get_table_metadata  # Importing the function
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from logger_config import configure_logging, log_execution_time
import logging
# Configure logging
load_dotenv()  # Load environment variables from .env file
# Initialize logging before creating the app
configure_logging()

# Create main application logger
logger = logging.getLogger("app")
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
# Set up static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

# Initialize the BlobServiceClient
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    logger.info("Azure Blob Initialised for FAQs")
except Exception as e:
    logger.error(f"Error initializing BlobServiceClient: {e}")
    # Handle the error appropriately, possibly exiting the application
    raise  # Re-raise the exception to prevent the app from starting
from pydantic import BaseModel
class ChartRequest(BaseModel):
    """
    Pydantic model for chart generation requests.
    """
    table_name: str
    x_axis: str
    y_axis: str
    chart_type: str

    class Config:  # This ensures compatibility with FastAPI
        json_schema_extra = {
            "example": {
                "table_name": "example_table",
                "x_axis": "column1",
                "y_axis": "column2",
                "chart_type": "Line Chart"
            }
        }

# Initialize OpenAI API key and model
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get('AZURE_DEPLOYMENT_NAME')

# Initialize the Azure OpenAI client
azure_openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

llm = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0
)

databases = ["Azure SQL"]
question_dropdown = os.getenv('Question_dropdown')

if 'messages' not in session_state:
    session_state['messages'] = []

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

def download_as_excel(data: pd.DataFrame, filename: str = "data.xlsx"):
    """
    Converts a Pandas DataFrame to an Excel file and returns it as a stream.

    Args:
        data (pd.DataFrame): The DataFrame to convert.
        filename (str): The name of the Excel file.  Defaults to "data.xlsx".

    Returns:
        BytesIO:  A BytesIO object containing the Excel file.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)  # Reset the pointer to the beginning of the stream
    return output

def create_gauge_chart_json(title, value, min_val=0, max_val=100, color="blue", subtext="%"):
    """
    Creates a gauge chart using Plotly and returns it as a JSON string.

    Args:
        title (str): The title of the chart.
        value (float): The value to display on the gauge.
        min_val (int): The minimum value of the gauge.  Defaults to 0.
        max_val (int): The maximum value of the gauge.  Defaults to 100.
        color (str): The color of the gauge.  Defaults to "blue".
        subtext (str): The subtext to display below the value. Defaults to "%".

    Returns:
        str: A JSON string representation of the gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': 'black'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 1},
            'bgcolor': "white",
            'borderwidth': 0.7,
            'bordercolor': "black",

            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': subtext, 'font': {'size': 16, 'color': 'gray'}}
    ))

    # Adjust the layout to prevent cropping
    fig.update_layout(
        width=350,  # Increased width
        height=350,  # Increased height
        margin=dict(
            t=50,  # Top margin
            b=50,  # Bottom margin
            l=50,  # Left margin
            r=50   # Right margin
        )

    )
    return fig.to_json()  # Return JSON instead of an image

@app.get("/get-table-columns/")
async def get_table_columns(table_name: str):
    """
    Fetches the columns for a given table from the session state.

    Args:
        table_name (str): The name of the table.

    Returns:
        JSONResponse: A JSON response containing the list of columns or an error message.
    """
    try:
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found in session.")

        # Retrieve the DataFrame for the specified table
        data_df = session_state["tables_data"][table_name]
        columns = list(data_df.columns)

        return {"columns": columns}
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error fetching columns: {str(e)}"},
            status_code=500
        )

class QueryInput(BaseModel):
    """
    Pydantic model for user query input.
    """
    query: str

@app.post("/add_to_faqs")
async def add_to_faqs(data: QueryInput, subject:str, request: Request):
    """
    Adds a user query to the FAQ CSV file on Azure Blob Storage.

    Args:
        data (QueryInput): The user query.

    Returns:
        JSONResponse: A JSON response indicating success or failure.
    """
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Invalid query!")
    question_type = request.session.get('current_question_type')

    if question_type == 'generic':
        blob_name = f'table_files/{subject}_questions_generic.csv'
    elif question_type == "usecase":
        blob_name = f'table_files/{subject}_questions.csv'
    try:
        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)

        try:
            # Download the blob content
            blob_content = blob_client.download_blob().content_as_text()
        except ResourceNotFoundError:
            # If the blob doesn't exist, create a new one with a header if needed
            blob_content = "question\n"  # Replace with your actual header

        # Append the new query to the existing CSV content
        updated_csv_content = blob_content + f"{query}\n"  # Append new query

        # Upload the updated CSV content back to Azure Blob Storage
        blob_client.upload_blob(updated_csv_content.encode('utf-8'), overwrite=True)

        return {"message": "Query added to FAQs successfully and uploaded to Azure Blob Storage!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def generate_chart_figure(data_df: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str):
    """
    Generates a Plotly figure based on the specified chart type.
    Includes support for Word Cloud visualization.

    Args:
        data_df (pd.DataFrame): The DataFrame containing the data.
        x_axis (str): The column name for the x-axis.
        y_axis (str): The column name for the y-axis.
        chart_type (str): The type of chart to generate.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure, or None if the chart type is unsupported.
    """
    fig = None
    try:
        if chart_type == "Line Chart":
            fig = px.line(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Bar Chart":
            fig = px.bar(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Pie Chart":
            fig = px.pie(data_df, names=x_axis, values=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Box Plot":
            fig = px.box(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Heatmap":
            fig = px.density_heatmap(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Violin Plot":
            fig = px.violin(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Area Chart":
            fig = px.area(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Funnel Chart":
            fig = px.funnel(data_df, x=x_axis, y=y_axis)
        elif chart_type == "Word Cloud":
            # For Word Cloud, we only need text data from x_axis column
            text_data = data_df[x_axis].dropna().astype(str).tolist()
            text = ' '.join(text_data)
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                max_words=200).generate(text)
            
            # Convert to Plotly figure
            fig = px.imshow(wordcloud.to_array())
            fig.update_layout(
                
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
        return fig
    except Exception as e:
        logger.error(f"Error generating {chart_type} chart: {str(e)}")
        raise

class ChartRequest(BaseModel):
    table_name: str
    x_axis: str
    y_axis: str
    chart_type: str

@app.post("/generate-chart")
async def generate_chart(request: ChartRequest):
    """
    Generates a chart based on the provided request data.
    Handles both numeric charts and text-based Word Cloud.
    """
    logger.info(f"Endpoint: /generate-chart - Chart request: {request.dict()}")
    with log_execution_time("generate_chart"):
        try:
            table_name = request.table_name
            x_axis = request.x_axis
            y_axis = request.y_axis
            chart_type = request.chart_type


            # Validate table exists
            if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
                raise HTTPException(status_code=404, detail=f"No data found for table {table_name}")

            data_df = session_state["tables_data"][table_name]
            
            # Validate columns exist
            if x_axis not in data_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{x_axis}' not found in data")
                
            # Skip y_axis validation for Word Cloud
            if chart_type != "Word Cloud" and y_axis not in data_df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{y_axis}' not found in data")

            # Data processing based on chart type
            if chart_type == "Word Cloud":
                # Ensure we have text data for word cloud
                if not pd.api.types.is_string_dtype(data_df[x_axis]):
                    data_df[x_axis] = data_df[x_axis].astype(str)
            else:
                # For other charts, convert y-axis to numeric
                try:
                    data_df[y_axis] = pd.to_numeric(data_df[y_axis], errors='coerce')
                    data_df = data_df.dropna(subset=[y_axis])
                    if len(data_df) == 0:
                        raise ValueError("No valid numeric data available after conversion")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing numeric data: {str(e)}")

            # Generate the chart
            fig = generate_chart_figure(data_df, x_axis, y_axis, chart_type)
            
            if fig is None:
                raise HTTPException(status_code=400, detail="Unsupported chart type selected")
            logger.info(f"Generating {request.chart_type} chart")
            return JSONResponse(content={"chart": fig.to_json()})

        except HTTPException as he:

            raise he
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}", exc_info=True)

            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/download-table/")
@app.get("/download-table")
async def download_table(table_name: str):
    """
    Downloads a table as an Excel file.

    Args:
        table_name (str): The name of the table to download.

    Returns:
        StreamingResponse: A streaming response containing the Excel file.
    """
    logger.info("Endpoint: /download_table ")
    with log_execution_time("download-table"):
        try:
            # Check if the requested table exists in session state
            if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
                raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

            # Get the table data from session_state
            data = session_state["tables_data"][table_name]

            # Generate Excel file
            output = download_as_excel(data, filename=f"{table_name}.xlsx")

            # Return the Excel file as a streaming response
            response = StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            response.headers["Content-Disposition"] = f"attachment; filename={table_name}.xlsx"
            return response

        except Exception as e:
            logger.error(f"error while downloading table: {e}")
# Replace APIRouter with direct app.post
def format_number(x):
    if isinstance(x, int):  # Check if x is an integer
        return f"{x:d}"
    elif isinstance(x, float) and x.is_integer():  # Check if x is a float and is equivalent to an integer
        return f"{int(x):d}"
    else:
        return f"{x:.1f}"  # For other floats, format with one decimal place
@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file using Azure OpenAI's Whisper model.

    Args:
        file (UploadFile): The audio file to transcribe.

    Returns:
        JSONResponse: A JSON response containing the transcription or an error message.
    """
    logger.info("Audio transcription started")
    try:
        # Check if API key is available
        if not AZURE_OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="Missing Azure OpenAI API Key.")
        
        # Read audio file
        audio_bytes = await file.read()
        audio_bio = BytesIO(audio_bytes)
        audio_bio.name = file.filename  # Use original filename or set appropriate extension

        # Transcribe using Azure OpenAI
        transcript = azure_openai_client.audio.transcriptions.create(
            model="whisper-1",  # Azure deployment name for Whisper model
            file=audio_bio
        )

        return {"transcription": transcript.text}

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error transcribing audio: {str(e)}"}, 
            status_code=500
        )

@app.get("/get_questions/")
@app.get("/get_questions")
async def get_questions(subject: str, request: Request):
    """
    Fetches questions from a CSV file in Azure Blob Storage based on the selected subject.

    Args:
        subject (str): The subject to fetch questions for.

    Returns:
        JSONResponse: A JSON response containing the list of questions or an error message.
    """
    question_type = request.session.get('current_question_type')
    if question_type == 'generic':
        csv_file_name = f"table_files/{subject}_questions_generic.csv"
    else: 
        csv_file_name = f"table_files/{subject}_questions.csv"
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=csv_file_name)

    try:
        # Check if the blob exists
        if not blob_client.exists():
            logger.error(f"file not found {csv_file_name}")
            return JSONResponse(
                content={"error": f"The file {csv_file_name} does not exist."}, status_code=404
            )

        # Download the blob content
        blob_content = blob_client.download_blob().content_as_text()

        # Read the CSV content
        questions_df = pd.read_csv(StringIO(blob_content))
        
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()

        return {"questions": questions}

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )
# Function to load prompts from YAML

def load_prompts(filename:str):
    """
    Loads prompts from the chatbot_prompt.yaml file.

    Returns:
        dict: A dictionary containing the loaded prompts.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error reading prompts file: {e}")
        return {}
    
         
# @app.post("/submit_feedback/")
# @app.post("/submit_feedback")
# async def submit_feedback(request: Request):
#     data = await request.json() # Corrected for FastAPI
    
#     table_name = data.get("table_name")
#     feedback_type = data.get("feedback_type")
#     user_query = data.get("user_query")
#     sql_query = data.get("sql_query")

#     if not table_name or not feedback_type:
#         return JSONResponse(content={"success": False, "message": "Table name and feedback type are required."}, status_code=400)

#     try:
#         # Create database connection
#         engine = create_engine(
#         f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}'
#         )
#         Session = sessionmaker(bind=engine)
#         session = Session()

#         # Sanitize input (Escape single quotes)
#         table_name = escape_single_quotes(table_name)
#         user_query = escape_single_quotes(user_query)
#         sql_query = escape_single_quotes(sql_query)
#         feedback_type = escape_single_quotes(feedback_type)

#         # Insert feedback into database
#         insert_query = f"""
#         INSERT INTO lz_feedbacks (department, user_query, sql_query, table_name, data, feedback_type, feedback)
#         VALUES ('unknown', :user_query, :sql_query, :table_name, 'no data', :feedback_type, 'user feedback')
#         """

#         session.execute(insert_query, {
#         "table_name": table_name,
#         "user_query": user_query,
#         "sql_query": sql_query,
#         "feedback_type": feedback_type
#         })

#         session.commit()
#         session.close()

#         return JSONResponse(content={"success": True, "message": "Feedback submitted successfully!"})

#     except Exception as e:
#         session.rollback()
#         session.close()
#         return JSONResponse(content={"success": False, "message": f"Error submitting feedback: {str(e)}"}, status_code=500)


import csv

def get_keyphrases():
    keyphrases = []
    with open('table_files/keyphrases_rephrasing.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assumes the column is named exactly 'keyphrases'
            if 'keyphrases' in row and row['keyphrases']:
                keyphrases.append(row['keyphrases'])
    return ','.join(keyphrases)

if 'messages' not in session_state:
    session_state['messages'] = []
    
def parse_table_data(csv_file_path):
    """
    Parses a CSV file containing table definitions and returns structured data.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        dict: A dictionary with table names as keys and their metadata as values
              Format: {
                  'table_name': {
                      'description': 'table description',
                      'columns': [
                          {
                              'name': 'column_name',
                              'type': 'data_type',
                              'nullable': boolean,
                              'description': 'column description'
                          },
                          ...
                      ]
                  },
                  ...
              }
    """
    tables = defaultdict(lambda: {
        'description': '',
        'columns': []
    })
    
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if len(row) < 3:  # Skip incomplete rows
                continue
                
            table_name = row[0].strip()
            table_description = row[1].strip()
            column_info = row[2].strip()
            
            # Parse column information (name, type, nullable, description)
            if '(' in column_info:
                # Extract column name and type
                col_name = column_info.split('(')[0].strip()
                type_part = column_info.split('(')[1].split(')')[0].strip()
                
                # Check for NULLABLE
                nullable = 'NULLABLE' in column_info
                
                # Extract description (after colon if present)
                if ':' in column_info:
                    col_desc = column_info.split(':')[-1].strip()
                else:
                    col_desc = ''
            else:
                col_name = column_info
                type_part = ''
                nullable = False
                col_desc = ''
            
            # Ensure table exists in dictionary
            if table_name not in tables:
                tables[table_name]['description'] = table_description
            
            # Add column information
            tables[table_name]['columns'].append({
                'name': col_name,
                'type': type_part,
                'nullable': nullable,
                'description': col_desc
            })
    
    return dict(tables)


@app.post("/submit")
async def submit_query(
    request: Request,
    section: str = Form(...),
    database: str = Form(...), 
    user_query: str = Form(...),
    page: int = Query(1),
    records_per_page: int = Query(10),
    model: Optional[str] = Form(AZURE_DEPLOYMENT_NAME)
):
    logger.info(f"Endpoint: /submit - Received query: {user_query}, Section: {section}")
    with log_execution_time("submit_query"):    
    # Initialize response structure
        response_data = {
            "user_query": user_query,
            "query": "",
            "tables": [],
            "llm_response": "",
            "chat_response": "",
            "history": session_state.get('messages', []),
            "interprompt": "",
            "langprompt": "",
            "error": None
        }

        try:
            # Reset per-request variables
            unified_prompt = ""
            final_prompt = ""
            llm_reframed_query = ""
            
            # Get current question type from session
            current_question_type = request.session.get("current_question_type", "generic")
            prompts = request.session.get("prompts", load_prompts("generic_prompt.yaml"))
            logger.debug(f"Current question type: {request.session.get('current_question_type')}")
            # Handle session messages
            if 'messages' not in session_state:
                session_state['messages'] = []
            
            session_state['messages'].append({"role": "user", "content": user_query})
            chat_history = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in session_state['messages'][-10:]
            )

            # Step 1: Generate unified prompt based on question type
            try:
                if current_question_type == "usecase":
                    key_parameters = get_key_parameters()
                    keyphrases = get_keyphrases()
                    unified_prompt = prompts["unified_prompt"].format(
                        user_query=user_query,
                        chat_history=chat_history,
                        key_parameters=key_parameters,
                        keyphrases=keyphrases
                    )
                    logger.info("Generating rephrased query")

                    llm_reframed_query = llm.invoke(unified_prompt).content.strip()
                    intent_result = intent_classification(llm_reframed_query)
                    logger.info("Intent Classification Result: ", intent_result)
                    if not intent_result:
                        raise HTTPException(
                            status_code=400,
                            detail="Please rephrase or add more details to your question"
                        )
                    
                    chosen_tables = intent_result["tables"]
                    logger.info(f"Chosen tables: {chosen_tables}")
                    selected_business_rule = get_business_rule(intent_result["intent"])
                    logger.info("Business rules added")
                elif current_question_type == "generic":
                    tables_metadata = get_table_metadata()
                    unified_prompt = prompts["unified_prompt"].format( 
                        user_query=user_query,
                        chat_history=chat_history,
                        key_parameters=get_key_parameters(),
                        keyphrases=get_keyphrases(),
                        table_metadata=tables_metadata
                    )
                    
                    llm_response_str = llm.invoke(unified_prompt).content.strip()
                    try:
                        llm_result = json.loads(llm_response_str)
                        llm_reframed_query = llm_result.get("rephrased_query", "")
                        chosen_tables = ["mh_ad_ai_dimension","mh_model_master","mh_ro_hdr_details","mh_ro_parts","mh_ro_labour", "mh_cust_verbatim"]
                        logger.info(f"Chosen tables: {chosen_tables}")

                        selected_business_rule = ""
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to parse LLM response"
                        )
                
                response_data["llm_response"] = llm_reframed_query
                response_data["interprompt"] = unified_prompt
                logger.debug(f"Interprated Query: {llm_reframed_query}")
            except Exception as e:
                logger.error(f"Prompt generation error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Prompt generation failed: {str(e)}"
                )

            # Step 2: Invoke LangChain
            try:
                relationships = find_relationships_for_tables(chosen_tables, 'table_relation.json')
                table_details = get_table_details(table_name=chosen_tables)
                logger.info("Table relationships added, calling invoke chain")
                response, chosen_tables, tables_data, agent_executor, final_prompt = invoke_chain(
                    llm_reframed_query,
                    session_state['messages'],
                    model,
                    section,
                    database,
                    table_details,
                    selected_business_rule,
                    current_question_type,
                    relationships
                )
                logger.debug(f"LangChain response: {response}")
                response_data["langprompt"] = str(final_prompt)
                
                if isinstance(response, str):
                    response_data["query"] = response
                    session_state['generated_query'] = response
                else:
                    response_data["query"] = response.get("query", "")
                    session_state['generated_query'] = response.get("query", "")
                    session_state['chosen_tables'] = chosen_tables
                    session_state['tables_data'] = tables_data

            except Exception as e:
                logger.error(f"LangChain invocation error: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Query execution failed: {str(e)}"
                )
            logger.info(f"Processed {len(chosen_tables)} tables")

            # Step 3: Process results
            if chosen_tables and 'tables_data' in session_state:
                try:
                    # Format numeric columns
                    for table_name, df in session_state['tables_data'].items():
                        for col in df.select_dtypes(include=['number']).columns:
                            session_state['tables_data'][table_name][col] = df[col].apply(format_number)
                    
                    # Prepare table HTML
                    response_data["tables"] = prepare_table_html(
                        session_state['tables_data'],
                        page,
                        records_per_page
                    )
                    
                    # Generate insights if data exists
                    # data_preview = next(iter(session_state['tables_data'].values())).head(5).to_string(index=False)
                    response_data["chat_response"] = ""  # Placeholder for actual insights
                    
                except Exception as e:
                    logger.error(f"Data processing error: {str(e)}")
                    response_data["chat_response"] = f"Data retrieved but processing failed: {str(e)}"

            # Append successful response to chat history
            session_state['messages'].append({
                "role": "assistant",
                "content": response_data["chat_response"]
            })

            return JSONResponse(content=response_data)

        except HTTPException as he:
            logger.error(f"Error in submit_query: {he.detail}", exc_info=True)

            # Capture error details
            response_data.update({
                "chat_response": f"Error: {he.detail}",
                "error": str(he.detail),
                "history": session_state.get('messages', []),
                "langprompt": str(final_prompt) if 'final_prompt' in locals() else "Not generated due to error",
                "interprompt": unified_prompt if 'unified_prompt' in locals() else "Not generated due to error"
            })
            
            session_state['messages'].append({
                "role": "assistant",
                "content": f"Error: {he.detail}"
            })
            
            return JSONResponse(
                content=response_data,
                status_code=he.status_code
            )
            
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            response_data.update({
                "chat_response": "An unexpected error occurred",
                "error": str(e),
                "history": session_state.get('messages', []),
                "langprompt": str(final_prompt) if 'final_prompt' in locals() else "Not generated due to error",
                "interprompt": unified_prompt if 'unified_prompt' in locals() else "Not generated due to error"
            })
            
            session_state['messages'].append({
                "role": "assistant",
                "content": "An unexpected error occurred"
            })
            
            return JSONResponse(
                content=response_data,
                status_code=500
            )

# Replace APIRouter with direct app.post

@app.post("/reset-session")
async def reset_session(request: Request):
    """
    Resets the session state by clearing the session_state dictionary.
    """
    global session_state
    with session_lock:
        session_state.clear()
        session_state['messages'] = []
    # Reset per-user session variables
    request.session.clear()
    request.session["current_question_type"] = "generic"
    request.session["prompts"] = load_prompts("generic_prompt.yaml")
    return {"message": "Session state cleared successfully"}, 200

def prepare_table_html(tables_data, page, records_per_page):
    """
    Prepares HTML for displaying table data with pagination.

    Args:
        tables_data (dict): A dictionary of table names and their corresponding DataFrames.
        page (int): The current page number.
        records_per_page (int): The number of records to display per page.

    Returns:
        list: A list of dictionaries containing table name, HTML, and pagination information.
    """
    tables_html = []
    for table_name, data in tables_data.items():
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page
        html_table = display_table_with_styles(data, table_name, page, records_per_page)
        logger.debug("Returned table data",exc_info=1)
        tables_html.append({
            "table_name": table_name,
            "table_html": html_table,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "records_per_page": records_per_page,
            }
        })
    return tables_html

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the root HTML page.

    Args:
        request (Request): The incoming request.

    Returns:
        TemplateResponse: The rendered HTML template.
    """
    logger.info("Endpoint: / - Loading root page")
    with log_execution_time("read_root"):
        try:
    # Extract table names dynamically
            logger.debug("Extracting table names")

            tables = []
            if "prompts" in request.session:
                del request.session["prompts"]
            # Only set defaults if not already set
            if "current_question_type" not in request.session:
                request.session["current_question_type"] = "generic"
                request.session["prompts"] = load_prompts("generic_prompt.yaml")

            # Pass dynamically populated dropdown options to the template
            return templates.TemplateResponse("index.html", {
                "request": request,
                "databases": databases,                                     
                "tables": tables,        # Table dropdown based on database selection
                "question_dropdown": question_dropdown.split(','),  # Static questions from env
            })
        except Exception as e:
            logger.error(f"Error in read_root: {str(e)}", exc_info=True)

# Table data display endpoint
def display_table_with_styles(data, table_name, page_number, records_per_page):
    """
    Displays a Pandas DataFrame as an HTML table with custom styles and pagination.

    Args:
        data (pd.DataFrame): The DataFrame to display.
        table_name (str): The name of the table.
        page_number (int): The current page number.
        records_per_page (int): The number of records to display per page.

    Returns:
        str: An HTML string representation of the styled table.
    """
    start_index = (page_number - 1) * records_per_page
    end_index = start_index + records_per_page
    page_data = data.iloc[start_index:end_index]
    # Ensure that the index always starts from 1 for each page
    page_data.index = range(start_index + 1, start_index + 1 + len(page_data))
    styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
        .set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#333'),
                          ('color', 'white')]
            },
                {
                    'selector': 'td',
                    'props': [('border', '1px solid black')]
                }
            ])
    
    return styled_table.to_html()


@app.get("/get_table_data/")
@app.get("/get_table_data")
async def get_table_data(
    table_name: str = Query(...),
    page_number: int = Query(1),
    records_per_page: int = Query(10),
):
    """Fetch paginated and styled table data."""
    try:
        # Check if the requested table exists in session state
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

        # Retrieve the data for the specified table
        data = session_state["tables_data"][table_name]
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        # Ensure valid page number
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(status_code=400, detail="Invalid page number.")

        # Slice data for the requested page
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]

        # Style the table as HTML
        styled_table = (
            page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
            ])
            .to_html(escape=False)  # Render as HTML
        )

        return {
            "table_html": styled_table,
            "page_number": page_number,
            "total_pages": total_pages,
            "total_records": total_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")


class QuestionTypeRequest(BaseModel):
    question_type: str
@app.post("/set-question-type")
async def set_question_type(payload: QuestionTypeRequest, request: Request):
    current_question_type = payload.question_type
    filename = "generic_prompt.yaml" if current_question_type == "generic" else "chatbot_prompt.yaml"
    # To force reload, remove the session key before loading
    

    prompts = load_prompts(filename)
    request.session["current_question_type"] = current_question_type
    request.session["prompts"] = prompts  # If you want to store prompts per session

    return JSONResponse(content={"message": "Question type set", "prompts": prompts})

