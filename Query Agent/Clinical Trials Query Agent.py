# Databricks notebook source
# MAGIC %md
# MAGIC DATABASE_URI = "databricks+connector://token:dapi74b032fd2dff411d17f65b5de5eeb347@dbc-6859411e-562c.cloud.databricks.com:443/clinicaltrials?http_path=/sql/1.0/warehouses/2b225a04a2c0109b"

# COMMAND ----------

# DBTITLE 1,installs
# MAGIC %pip install databricks-langchain langchain databricks-sql-connector
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,imports
from databricks_langchain import ChatDatabricks
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from databricks import sql
import re

# COMMAND ----------

# DBTITLE 1,functions
# SQL extraction helper
def extract_sql(text: str) -> str:
    match = re.search(r"```sql\\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    for line in lines:
        if line.strip().lower().startswith("select"):
            return "\n".join(lines[lines.index(line):]).strip()
    return ""
    
def run_query_from_nl_v2(question: str) -> str:
    print("Received question:", question)
    llm_output = chain.run(question)
    print("LLM output:\n", llm_output)

    sql_query = extract_sql(llm_output)
    print("Extracted SQL:\n", sql_query)

    if not sql_query.lower().startswith("select"):
        return "Failed to extract valid SQL query."

    try:
        conn = sql.connect(
            server_hostname="dbc-6859411e-562c.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/2b225a04a2c0109b",
            access_token="dapi74b032fd2dff411d17f65b5de5eeb347"
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchmany(10)
        return "\n".join(str(row) for row in rows) or "No results found."
    except Exception as e:
        return f"SQL execution failed: {e}"

# COMMAND ----------

llm = ChatDatabricks(endpoint="/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/", temperature=0)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert SQL generator.

Table: clinicaltrials.default.ctg_studies
Columns: `NCT Number`, `Study Title`, `Study URL`, `Study Status`, `Brief Summary`, `Conditions`, `Interventions`,
         `Primary Outcome Measures`, `Secondary Outcome Measures`, `Sponsor`, `Collaborators`, `Sex`, `Age`,
         `Phases`, `Enrollment`, `Study Type`, `Start Date`, `Locations`

Always quote column names using backticks (`) if they contain spaces.

Here are the valid values for `Study Status`:
ENROLLING_BY_INVITATION
NOT_YET_RECRUITING
UNKNOWN
SUSPENDED
TERMINATED
NO_LONGER_AVAILABLE
RECRUITING
APPROVED_FOR_MARKETING
WITHDRAWN
COMPLETED
ACTIVE_NOT_RECRUITING
AVAILABLE
TEMPORARILY_NOT_AVAILABLE

`Interventions` are typically prefixed with the following values, followed by a colon:
DEVICE
DRUG
BEHAVIORAL
RADIATION
PROCEDURE
DIAGNOSTIC_TEST
BIOLOGICAL
OTHER
COMBINATION_PRODUCT
DIETARY_SUPPLEMENT
GENETIC

Here are the valid values for `Age`:
ADULT
ADULT, OLDER_ADULT
CHILD, ADULT, OLDER_ADULT
CHILD
CHILD, ADULT
OLDER_ADULT

Here are the valud values for `Sex`:
MALE
FEMALE
ALL

Here are the valid values for `Phases`:
PHASE1
PHASE2
NA
PHASE2|PHASE3
PHASE1|PHASE2
PHASE4
EARLY_PHASE1
PHASE3

When taking user input use the lowercase of the input and compare it to table columns using the lowercase functions.

For each result, we will add a computed column, `Trial Quality Score` using the following:
  When the `Study Status` is 'RECRUITING' then add 0.4 to the score
  When the `Phases` is 'PHASE3' or 'PHASE4' then add 0.2 to the score
  When the `Start Date` is within 2 years of today then add 0.2 to the score
  When the `Enrollment` is great then 100 then at 0.2 to the score

You should calculate the distance between San Francisco, California, and the items `Locations` that are separated by a "|" and put this in a column called `Distance`. The `Locations` column contains location names, not latitude and longitude coordinated.  You should make a best attempt at calculating the distance based on your knowledge, 
and no other data.

Return the results order by highest `Trial Quality Score` and least `Distance`

User Question: {question}
SQL:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

# STEP 4: Tool and Agent
tool = Tool(
    name="ClinicalTrialsNLQuery",
    func=run_query_from_nl_v2,
    description="Ask natural questions to query clinical_trials."
)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

print(run_query_from_nl_v2("What lung cancer trials are currently enrolling participants?"))

# Example runner
#if __name__ == "__main__":
#    print(agent.run("List lung cancer trials with available enrollment"))


# COMMAND ----------

# File: deploy_clinical_trials_agent.py

import os
import re
import mlflow
import mlflow.pyfunc
from databricks import sql
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks
from mlflow.models.signature import infer_signature
import pandas as pd

# Environment variables (replace with secure values or dbutils.secrets)
os.environ["DATABRICKS_SERVER_HOSTNAME"] = "dbc-6859411e-562c.cloud.databricks.com"
os.environ["DATABRICKS_HTTP_PATH"] = "/sql/1.0/warehouses/2b225a04a2c0109b"
os.environ["DATABRICKS_TOKEN"] = "dapi74b032fd2dff411d17f65b5de5eeb347"
USER_NAME = "tom@keywell.ai"  # <-- update this
EXPERIMENT_PATH = f"/Users/{USER_NAME}/clinical_trials_agent"
REGISTERED_MODEL_NAME = "clinical_trials_nl_agent"

# Initialize LLM
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0)

# Prompt for generating SQL from natural language
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert SQL generator.

Table: clinicaltrials.default.ctg_studies
Columns: `NCT Number`, `Study Title`, `Study URL`, `Study Status`, `Brief Summary`, `Conditions`, `Interventions`,
         `Primary Outcome Measures`, `Secondary Outcome Measures`, `Sponsor`, `Collaborators`, `Sex`, `Age`,
         `Phases`, `Enrollment`, `Study Type`, `Start Date`, `Locations`

Always quote column names using backticks (`) if they contain spaces.

Here are the valid values for `Study Status`:
ENROLLING_BY_INVITATION
NOT_YET_RECRUITING
UNKNOWN
SUSPENDED
TERMINATED
NO_LONGER_AVAILABLE
RECRUITING
APPROVED_FOR_MARKETING
WITHDRAWN
COMPLETED
ACTIVE_NOT_RECRUITING
AVAILABLE
TEMPORARILY_NOT_AVAILABLE

`Interventions` are typically prefixed with the following values, followed by a colon:
DEVICE
DRUG
BEHAVIORAL
RADIATION
PROCEDURE
DIAGNOSTIC_TEST
BIOLOGICAL
OTHER
COMBINATION_PRODUCT
DIETARY_SUPPLEMENT
GENETIC

Here are the valid values for `Age`:
ADULT
ADULT, OLDER_ADULT
CHILD, ADULT, OLDER_ADULT
CHILD
CHILD, ADULT
OLDER_ADULT

Here are the valud values for `Sex`:
MALE
FEMALE
ALL

Here are the valid values for `Phases`:
PHASE1
PHASE2
NA
PHASE2|PHASE3
PHASE1|PHASE2
PHASE4
EARLY_PHASE1
PHASE3

When taking user input use the lowercase of the input and compare it to table columns using the lowercase functions.

For each result, we will add a computed column, `Trial Quality Score` using the following:
  When the `Study Status` is 'RECRUITING' then add 0.4 to the score
  When the `Phases` is 'PHASE3' or 'PHASE4' then add 0.2 to the score
  When the `Start Date` is within 2 years of today then add 0.2 to the score
  When the `Enrollment` is great then 100 then at 0.2 to the score

Return the results order by highest `Trial Quality Score` 

User Question: {question}
SQL:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

# SQL extraction helper
def extract_sql(text: str) -> str:
    match = re.search(r"```sql\\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    for line in lines:
        if line.strip().lower().startswith("select"):
            return "\n".join(lines[lines.index(line):]).strip()
    return ""

# Query runner
def query_clinical_trials(question: str) -> str:
    llm_output = chain.run(question)
    sql_query = extract_sql(llm_output)
    if not sql_query.lower().startswith("select"):
        return "Invalid query generated."

    try:
        conn = sql.connect(
            server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"]
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchmany(10)
        return "\n".join(str(row) for row in rows) or "No results found."
    except Exception as e:
        return f"SQL execution failed: {e}"

class ClinicalTrialAgent(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        question = model_input[0] if isinstance(model_input, list) else model_input
        return query_clinical_trials(question)

# ---- MLflow Logging with Signature ----
mlflow.set_experiment(EXPERIMENT_PATH)

example_input = pd.DataFrame({"question": ["What lung cancer trials are currently enrolling?"]})
example_output = pd.Series([query_clinical_trials(example_input.iloc[0, 0])])
signature = infer_signature(example_input, example_output)

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="clinical_trials_agent",
        python_model=ClinicalTrialAgent(),
        signature=signature,
        input_example=example_input
    )
    model_uri = f"runs:/{run.info.run_id}/clinical_trials_agent"
    registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"✅ Model registered as '{REGISTERED_MODEL_NAME}' at URI: {model_uri}")


# COMMAND ----------

from typing import Any, Generator, Optional, Sequence, Union

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert SQL generator for Databricks SQL.

Use this schema:

**Table:** `clinicaltrials.default.ctg_studies`  
**Columns:**  
- `NCT Number`, `Study Title`, `Study URL`, `Study Status`, `Brief Summary`, `Conditions`, `Interventions`,  
  `Primary Outcome Measures`, `Secondary Outcome Measures`, `Sponsor`, `Collaborators`, `Sex`, `Age`,  
  `Phases`, `Enrollment`, `Study Type`, `Start Date`, `Locations`

**ALWAYS**:
- Use backticks (`) around **any** column name with spaces.
- Use only provided **valid values** when filtering by `Study Status`, `Phases`, `Sex`, `Age`, or `Interventions`.
- Convert user inputs and column values to lowercase for filtering using `LOWER()`.

**Valid values**:
- `Study Status`: ENROLLING_BY_INVITATION, NOT_YET_RECRUITING, UNKNOWN, SUSPENDED, TERMINATED, NO_LONGER_AVAILABLE, RECRUITING, APPROVED_FOR_MARKETING, WITHDRAWN, COMPLETED, ACTIVE_NOT_RECRUITING, AVAILABLE, TEMPORARILY_NOT_AVAILABLE  
- `Phases`: PHASE1, PHASE2, NA, PHASE2|PHASE3, PHASE1|PHASE2, PHASE4, EARLY_PHASE1, PHASE3  
- `Sex`: MALE, FEMALE, ALL  
- `Age`: ADULT, ADULT, OLDER_ADULT, CHILD, ADULT, OLDER_ADULT, CHILD, CHILD, ADULT, OLDER_ADULT  
- `Interventions`: DEVICE, DRUG, BEHAVIORAL, RADIATION, PROCEDURE, DIAGNOSTIC_TEST, BIOLOGICAL, OTHER, COMBINATION_PRODUCT, DIETARY_SUPPLEMENT, GENETIC

**Computed Column – Trial Quality Score** (as `Trial Quality Score`):  
Sum of the following rules:
- +0.4 if LOWER(`Study Status`) = 'recruiting'  
- +0.2 if LOWER(`Phases`) IN ('phase3', 'phase4')  
- +0.2 if `Start Date` is within the last 2 years  
- +0.2 if `Enrollment` > 100

**Sort the final results** by `Trial Quality Score` descending.

Return only **executable SQL** with no explanation, markdown, or comments. Begin with `SELECT`.

User Question: {question}
SQL:
"""
)

###############################################################################
# Tools setup (if any needed in future)
###############################################################################
tools = []
uc_tool_names = ["system.ai.python_exec"]
uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
tools.extend(uc_toolkit.tools)

#####################
# Define agent logic
#####################

def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    def should_continue(state: ChatAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(state: ChatAgentState, config: RunnableConfig):
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(ChatAgentMessage(**msg) for msg in node_data.get("messages", []))
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"])


# Register the agent
mlflow.langchain.autolog()
compiled = create_tool_calling_agent(llm, tools, prompt.template)
AGENT = LangGraphChatAgent(compiled)
mlflow.models.set_model(AGENT)
