# Databricks notebook source
# DBTITLE 1,installs
# MAGIC %pip install databricks-langchain mlflow langchain langchain-core langgraph databricks-sql-connector
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,imports
# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC import mlflow
# MAGIC import re
# MAGIC from databricks_langchain import ChatDatabricks, UCFunctionToolkit
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langchain.tools import Tool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentResponse, ChatContext
# MAGIC from databricks import sql
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC # ============ CONFIG ============
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC WAREHOUSE_HOST = "dbc-6859411e-562c.cloud.databricks.com"
# MAGIC HTTP_PATH = "/sql/1.0/warehouses/2b225a04a2c0109b"
# MAGIC TOKEN = "dapi74b032fd2dff411d17f65b5de5eeb347"
# MAGIC
# MAGIC # ============ TOOL DEFINITION ============
# MAGIC def extract_sql(text: str) -> str:
# MAGIC     match = re.search(r"```sql\\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
# MAGIC     if match:
# MAGIC         return match.group(1).strip()
# MAGIC     lines = text.splitlines()
# MAGIC     for line in lines:
# MAGIC         if line.strip().lower().startswith("select"):
# MAGIC             return "\n".join(lines[lines.index(line):]).strip()
# MAGIC     return ""
# MAGIC
# MAGIC def query_clinical_trials(sql_input: str) -> str:
# MAGIC     try:
# MAGIC         conn = sql.connect(
# MAGIC             server_hostname=WAREHOUSE_HOST,
# MAGIC             http_path=HTTP_PATH,
# MAGIC             access_token=TOKEN,
# MAGIC         )
# MAGIC         cursor = conn.cursor()
# MAGIC         cursor.execute(sql_input)
# MAGIC         rows = cursor.fetchmany(10)
# MAGIC         return "\n".join(str(row) for row in rows) or "No results."
# MAGIC     except Exception as e:
# MAGIC         return f"SQL execution failed: {e}"
# MAGIC
# MAGIC sql_tool = Tool(
# MAGIC     name="ClinicalTrialsSQLQuery",
# MAGIC     description="Executes SQL queries against clinicaltrials.default.ctg_studies. Input must be valid SQL.",
# MAGIC     func=query_clinical_trials,
# MAGIC )
# MAGIC
# MAGIC # ============ AGENT DEFINITION ============
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolNode, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledStateGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         return "continue" if messages[-1].get("tool_calls") else "end"
# MAGIC
# MAGIC     preprocessor = RunnableLambda(lambda state: ([{"role": "system", "content": system_prompt}] + state["messages"]) if system_prompt else state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(state: ChatAgentState, config: RunnableConfig):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC system_prompt = """You are an expert SQL generator.
# MAGIC
# MAGIC Table: clinicaltrials.default.ctg_studies
# MAGIC Columns: `NCT Number`, `Study Title`, `Study URL`, `Study Status`, `Brief Summary`, `Conditions`, `Interventions`,
# MAGIC          `Primary Outcome Measures`, `Secondary Outcome Measures`, `Sponsor`, `Collaborators`, `Sex`, `Age`,
# MAGIC          `Phases`, `Enrollment`, `Study Type`, `Start Date`, `Locations`
# MAGIC
# MAGIC Always quote column names using backticks (`) if they contain spaces.
# MAGIC
# MAGIC Here are the valid values for `Study Status`:
# MAGIC ENROLLING_BY_INVITATION, NOT_YET_RECRUITING, UNKNOWN, SUSPENDED, TERMINATED,
# MAGIC NO_LONGER_AVAILABLE, RECRUITING, APPROVED_FOR_MARKETING, WITHDRAWN,
# MAGIC COMPLETED, ACTIVE_NOT_RECRUITING, AVAILABLE, TEMPORARILY_NOT_AVAILABLE
# MAGIC
# MAGIC Here are the valid values for `Phases`: PHASE1, PHASE2, PHASE3, PHASE4, etc.
# MAGIC
# MAGIC Add computed `Trial Quality Score`:
# MAGIC   +0.4 if `Study Status` = 'RECRUITING'
# MAGIC   +0.2 if `Phases` = 'PHASE3' or 'PHASE4'
# MAGIC   +0.2 if `Start Date` is within 2 years of today
# MAGIC   +0.2 if `Enrollment` > 100
# MAGIC
# MAGIC Order results by highest `Trial Quality Score`
# MAGIC
# MAGIC Use lowercase matching for user input. Only return SQL.
# MAGIC """
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(self, messages: list[ChatAgentMessage], context: Optional[ChatContext] = None, custom_inputs: Optional[dict[str, Any]] = None) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         messages_out = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages_out.extend(ChatAgentMessage(**msg) for msg in node_data.get("messages", []))
# MAGIC         return ChatAgentResponse(messages=messages_out)
# MAGIC
# MAGIC     def predict_stream(self, messages: list[ChatAgentMessage], context: Optional[ChatContext] = None, custom_inputs: Optional[dict[str, Any]] = None) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"])
# MAGIC
# MAGIC # ============ REGISTER AGENT ============
# MAGIC mlflow.langchain.autolog()
# MAGIC agent = create_tool_calling_agent(llm, [sql_tool], system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC print("✅ Agent registered with MLflow.")

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "What lung cancer trials are currently enrolling participants?"}]})

# COMMAND ----------

import mlflow
from agent import LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from pkg_resources import get_distribution

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        extra_pip_requirements=[f"databricks-connect=={get_distribution('databricks-connect').version}"],
        resources=resources,
    )

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "clinicaltrials"
schema = "default"
model_name = "clinical_trials_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)
