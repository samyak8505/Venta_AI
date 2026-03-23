import os
import io
import urllib.parse
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
import re
import subprocess
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dotenv import load_dotenv
load_dotenv()  # reads .env in dev; on server we'll use System env vars

import os, urllib.parse
# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(page_title="SQL Q&A Chatbot", page_icon="🗄️", layout="wide")

with st.sidebar:
    st.title("📊 AI Analytics Chatbot")
    st.caption("Ask questions about your SQL database")

    # API Key


    api_key = os.getenv("OPENAI_API_KEY", "")
    model_name = "gpt-4o"
    temperature = 0.0

    st.markdown("---")
    st.markdown("**Sample questions:**")
    st.markdown("- Compare sales of August 2025 and September 2025")
    st.markdown("- Carry out the order analysis of 2025")
    st.markdown("- Compare monthly sales growth for this year versus last year.")
    st.markdown("- Which regions showed the highest increase in sales compared to the previous quarter")
    st.markdown("- Which customers reduced their order volume significantly this quarter?")
    st.markdown("- Which products have the highest and lowest stock in hand?")
    st.markdown("- Which were the products with slow-moving stock in the past 3 months?")
    st.markdown("- 请显示过去六个月的销售趋势 (Show me the sales trend for the past six months)")
    st.markdown("- 销售最多的产品是哪个? (Which product has the highest sales?)")


st.title("📊 AI Analytics Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# HARDCODED DB CONNECTION
# ---------------------------
if not api_key:
    st.warning("⚠️ Enter your OpenAI API key in the sidebar.")
    st.stop()

# Hardcoded database details
user     = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host     = os.getenv("DB_HOST")
port     = int(os.getenv("DB_PORT", "1433"))
database = os.getenv("DB_NAME")

# List of allowed tables (replace with your actual table names)
# ---------------------------
# LIMIT ACCESS TO 5 VIEWS ONLY
# ---------------------------

ALLOWED_VIEWS = [
    "vw_Sales",
    "vw_Customers",
    "vw_Inventory",
    "vw_BatchBalance",
    "vw_OrderItem",
]

try:
    # Encode password safely
    encoded_password = urllib.parse.quote_plus(password)

    # SQLAlchemy connection string
    db_url = (
        f"mssql+pyodbc://{user}:{encoded_password}@{host}:{port}/{database}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
    )

    # ✅ Expose ONLY the 5 views to the agent
    db = SQLDatabase.from_uri(
    db_url,
    include_tables=ALLOWED_VIEWS,    # just the names
    sample_rows_in_table_info=3,
    schema="dbo",                    # ✅ important for SQL Server
    view_support=True                # ✅ include views in reflection
)

    st.success(f"✅ Connected to your Database ")

    # ---------- Schema display (views only) ----------

except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.info("Make sure you have the correct ODBC Driver 17 for SQL Server installed")
    st.stop()

llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

try:
    agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="tool-calling",      # ✅ critical change
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)
except Exception as e:
    st.error(f"Failed to create SQL agent: {e}")
    st.stop()    

def get_detailed_data_for_analysis(user_input, base_df, agent, db):
    """Based on the initial results, fetch additional detailed data for deeper analysis"""
    
    try:
        # Analyze what additional data would help explain the patterns
        detail_prompt = f"""
        Initial query: "{user_input}"
        Initial results (shape: {base_df.shape}, columns: {list(base_df.columns)}):
        {base_df.head().to_string()}
        
        What additional detailed data would help explain these patterns? 
        Return ONLY a SQL query to get customer-level or product-level details that might explain the trends.
        Focus on queries that would show which specific customers, products, or categories drove the changes.
        """
        
        detail_query_result = agent.invoke({"input": detail_prompt})
        detail_query = detail_query_result.get("output") if isinstance(detail_query_result, dict) else str(detail_query_result)
        detail_query = re.sub(r"```sql|```", "", detail_query).strip()
        
        if detail_query.lower().startswith("select"):
            # Get the detailed data
            detailed_df = pd.read_sql(detail_query, db._engine)
            return detailed_df
        else:
            return None
            
    except Exception as e:
        return None
    
def enhanced_data_driven_analysis(user_input, agent, db, original_df):
    """Perform deep, data-driven analysis with additional detailed data"""
    
    try:
        # Try to get detailed data for deeper analysis
        detailed_df = get_detailed_data_for_analysis(user_input, original_df, agent, db)
        
        if detailed_df is not None and len(detailed_df) > 0:
            analysis_prompt = f"""
            You are a data analyst. Analyze BOTH the summary data and detailed data to provide specific insights.

            USER QUESTION: "{user_input}"

            SUMMARY DATA (High-level results):
            {original_df.to_string()}
            
            DETAILED DATA (Customer/Product level):
            {detailed_df.head(20).to_string()}
            Detailed Data Shape: {detailed_df.shape}
            Detailed Columns: {list(detailed_df.columns)}
            
            Please analyze this data and provide:

            1. **ROOT CAUSE ANALYSIS**:
               - Which specific customers/products drove the changes?
               - Are there patterns in the detailed data that explain the summary trends?
               - Did any major customers stop ordering? Any products go out of stock?

            2. **SPECIFIC INSIGHTS**:
               - Reference actual customer names, product IDs, or amounts from the detailed data
               - Calculate percentage changes for specific items, not just totals

            3. **DATA-DRIVEN EXPLANATIONS**:
               - Don't guess generic reasons - use the actual data to explain patterns
               - If customer data shows specific drop-offs, mention them specifically

            Be extremely specific and data-driven.
            """
        else:
            # Fallback to original analysis if detailed data isn't available
            analysis_prompt = f"""
            You are a data analyst. Provide specific, data-driven insights.

            USER QUESTION: "{user_input}"

            DATA RETRIEVED:
            {original_df.to_string()}
            
            Data Shape: {original_df.shape}
            Data Columns: {list(original_df.columns)}
            
            Analyze this data and provide specific insights based on the actual numbers.
            """
        
        analysis_result = agent.invoke({"input": analysis_prompt})
        return analysis_result.get("output")
        
    except Exception as e:
        return f"Analysis error: {e}"


def perform_deep_dive_analysis(user_input, df, agent, db):
    """Perform additional data analysis to find specific patterns"""
    
    insights = []
    
    try:
        # Basic data analysis
        insights.append(f"**Data Overview**: {len(df)} records, {len(df.columns)} columns")
        
        # Analyze numeric columns for trends
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if df[col].notna().sum() > 0:
                    insights.append(f"**{col}**: Avg ${df[col].mean():.2f}, Max ${df[col].max():.2f}, Min ${df[col].min():.2f}")
        
        # Time-based analysis (if date columns exist)
        date_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
        
        if date_columns and len(df) > 1:
            date_col = date_columns[0]
            try:
                # Try to convert to datetime for time series analysis
                df_sorted = df.sort_values(date_col)
                if len(df_sorted) >= 2:
                    first_period = df_sorted.iloc[0]
                    last_period = df_sorted.iloc[-1]
                    
                    # Compare first and last period
                    for num_col in numeric_cols:
                        if num_col in first_period and num_col in last_period:
                            change = last_period[num_col] - first_period[num_col]
                            pct_change = (change / first_period[num_col]) * 100 if first_period[num_col] != 0 else 0
                            insights.append(f"**{num_col} Change**: {first_period[date_col]} → {last_period[date_col]}: ${change:+.2f} ({pct_change:+.1f}%)")
            except:
                pass
        
        # Top performers analysis
        for num_col in numeric_cols:
            if len(df) > 3:
                top_3 = df.nlargest(3, num_col)
                bottom_3 = df.nsmallest(3, num_col)
                
                insights.append(f"**Top 3 by {num_col}**: " + 
                               ", ".join([f"{row.get('product_name', row.get('customer_name', 'Item'))}: ${row[num_col]:.2f}" 
                                         for _, row in top_3.iterrows() if pd.notna(row[num_col])]))
        
        return "\n\n".join(insights)
        
    except Exception as e:
        return f"Deep dive analysis failed: {e}"

def get_related_data_queries(user_input, df, agent):
    """Suggest follow-up questions based on the current data"""
    
    prompt = f"""
    Based on this query: "{user_input}"
    And the data retrieved (shape: {df.shape}, columns: {list(df.columns)})
    
    Suggest 3 specific, data-driven follow-up questions that would help understand the patterns better.
    Focus on questions that can be answered by querying related tables.
    
    Format as bullet points.
    """
    
    try:
        suggestions = agent.invoke({"input": prompt})
        return suggestions.get("output")
    except:
        return "• What products contributed most to the sales change?\n• Which customers showed the biggest spending changes?\n• Are there regional patterns in the sales data?"

# ---------------------------
# QUERY HANDLER
# ---------------------------
def generate_why_queries(user_input, summary_df, db_schema, agent):
    prompt = f"""
    USER ASKED: "{user_input}"

    SUMMARY DATA:
    {summary_df.to_string(index=False)}

    DATABASE SCHEMA:
    {db_schema}

    TASK:
    - Suggest 2-3 SQL queries that would explain WHY this happened.
    - Focus on changes in products, customers, or regions.
    - Output JSON with fields: purpose, sql
    """
    response = agent.invoke({"input": prompt})
    try:
        return json.loads(response.get("output", "[]"))
    except:
        return []

def run_root_cause_v2(user_input, summary_df, agent, db):
    # get DB schema
    db_schema = db.get_table_info()

    # Step 1: Generate diagnostic queries
    queries = generate_why_queries(user_input, summary_df, db_schema, agent)

    results = {}
    for q in queries:
        try:
            sql = re.sub(r"```sql|```", "", q["sql"]).strip()
            results[q["purpose"]] = pd.read_sql(sql, db._engine)
        except Exception as e:
            results[q.get("purpose", "query")] = f"Failed: {e}"

    # Step 2: Ask LLM to synthesize WHY explanation
    reasoning_prompt = f"""
    USER ASKED: "{user_input}"

    SUMMARY DATA:
    {summary_df.to_string(index=False)}

    DIAGNOSTIC DATA:
    { {k: (v.to_string(index=False) if isinstance(v, pd.DataFrame) else v) for k,v in results.items()} }

    TASK:
    - Explain WHY the trend/change happened.
    - Refer to products, customers, or regions from the diagnostic data.
    - Be data-specific (use numbers/percentages).
    - Avoid speculation: base reasoning only on the provided data.
    """

    explanation = agent.invoke({"input": reasoning_prompt})
    return explanation.get("output")

def consolidated_analysis(user_input, df, agent):
    """One-shot analysis combining detailed insights, root cause, and follow-ups."""

    try:
        preview = df.head(20).to_string(index=False)
        columns = list(df.columns)
        schema_info = db.get_table_info()

        analysis_prompt = f"""
        You are a senior data analyst tasked with explaining business patterns
        in an e-commerce SQL database.

        USER QUESTION: "{user_input}"

        DATA PREVIEW (first 20 rows):
        {preview}

        DATA SHAPE: {df.shape}
        DATABASE SCHEMA: {schema_info}

        ### TASK
        Generate a structured, factual, and data oriented results based on the query and the data in **exactly three sections as given below**:
        
        1.**DETAILED INSIGHTS**:
        -Focus only on **numeric trends** and measurable results.
        - Summarize key metrics: averages, totals, growth/decline %, outliers, 
        - Identify patterns in numerical data (e.g., monthly sales up 12%, inventory down 8%).
         


        2.  **ROOT CAUSE ANALYSIS**:
            
          - Identify specific customers names, products names, regions names, or categories that drive the trends or anomalies.
          - INVESTIGATE THE PROBING QUESTIONS:
            If required join the tables to derive meaningful results or look into specific tables for below requirements.
            If further analysis of a particular table is required,use the database schema to carry it out and make sure its done.
            Remember all the data exists and is retrievable from the database so all the below questions must be strictly answered.
            
            Customers: Which specific customers(names) drove the changes? Did any major customers stop ordering, significantly reduce volume, or fail to renew or conversely, which customers demonstrated increased loyalty or a surge in spending??

            Products: Which specific products(names) drove the changes? Did any key products go out of stock, or got discontinued or if not this case which products were in high demand?

            Regions: Which specific regions drove the changes? Are there any geographic patterns (e.g., a single country, sales region) that explain the summary trend?

            Patterns: Find any other patterns that explain the summary trends..if no patterns are found give a generalised answer
        
        3. **Conclusion**:
          - Provide a strong comprehensive conclusion summarizing correlating the query and the detailed insights and the root cause analysis

        IMPORTANT:
        - Do not hallucinate
        - Be specific and data-driven.
        - Follow the 3-section structure exactly as written. 
        """

        result = agent.invoke({"input": analysis_prompt})
        return result.get("output") if isinstance(result, dict) else str(result)

    except Exception as e:
        return f"Analysis failed: {e}"


def handle_query(user_input, agent, db):
    try:
        # First try to get the SQL query and execute it directly (for Excel download)
        sql_result = agent.invoke({"input": f"Return only the SQL query (no explanation, no markdown) for: {user_input}"})
        sql_query = sql_result.get("output") if isinstance(sql_result, dict) else str(sql_result)
        sql_query = re.sub(r"```sql|```", "", sql_query).strip()

        if not sql_query.lower().startswith("select"):
            # For non-SELECT queries, use regular agent response
            result = agent.invoke({"input": user_input})
            return result.get("output") or result

        # Remove LIMIT for full data retrieval (for Excel download)
        sql_no_limit = re.sub(r"limit\s+\d+\s*;?", "", sql_query, flags=re.IGNORECASE).strip()
        
        # Get the actual data WITHOUT limit for full results
        df = pd.read_sql(sql_no_limit, db._engine)
        
        # Check if this is an analytical question
        analytical_keywords = ['compare', 'analysis', 'trend', 'why', 'reason', 'increase', 'decrease', 'growth', 'change', 'impact','pattern','which','what','how','analyze']
        is_analytical = any(keyword in user_input.lower() for keyword in analytical_keywords)
        
        if is_analytical and len(df) > 0:
            try:
                combined_analysis = consolidated_analysis(user_input, df, agent)
                return {
                    "data": df,
                    "analysis": combined_analysis,
                    "type": "consolidated_analytical"
                }
            except Exception as e:
                return {"data": df, "analysis": f"Consolidated analysis failed: {e}", "type": "consolidated_analytical"}
            
        return df

            
    except Exception as e:
        result = agent.invoke({"input": user_input})
        return result.get("output") or result
    

# ---------------------------
# CHAT UI
# ---------------------------
st.markdown("---")
st.subheader("Chat")

if st.button("🧹 Clear chat"):
    st.session_state.messages = []
    plt.close('all')
    st.experimental_rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "plot" in msg and msg["plot"]:
            st.pyplot(msg["plot"])
            plt.close(msg["plot"])
        st.markdown(msg["content"])
        if "dataframe" in msg and msg["dataframe"] is not None:
            st.dataframe(msg["dataframe"])
        if "download" in msg and msg["download"] is not None:
            st.download_button(**msg["download"])

user_input = st.chat_input("Ask a question about your database…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            plt.close('all')
            output = handle_query(user_input, agent, db)

            # Handle enhanced analytical results (dictionary output)
            if isinstance(output, dict) and output.get("type") == "consolidated_analytical":
                st.markdown("### 🔍 Analysis")
                st.markdown(output["analysis"])

                st.markdown("### 📋 Data Preview")
                df_full = output["data"]

                if len(df_full) > 10:
                    st.dataframe(df_full.head(10))
                    st.info(f"Showing 10 of {len(df_full)} records")

                    # 🔽 Add Excel download for the FULL result set
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        df_full.to_excel(writer, index=False, sheet_name="Results")

                    st.download_button(
                        label="📥 Download full results as Excel",
                        data=buffer.getvalue(),
                        file_name="query_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_{uuid.uuid4()}"
                    )
                else:
                    st.dataframe(df_full)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output["analysis"],
                    "dataframe": df_full.head(10),
                    "type": "consolidated_analytical"
                })

            # Handle DataFrame results (regular queries)
            elif isinstance(output, pd.DataFrame):
                df = output
                fig = None
                if plt.get_fignums():
                    fig = plt.gcf()

                if len(df) > 10:
                    placeholder.write("Showing first 10 records (download full results below):")
                    st.dataframe(df.head(10))

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="Results")

                    download_button = {
                        "label": "📥 Download full results as Excel",
                        "data": buffer.getvalue(),
                        "file_name": "query_results.xlsx",
                        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "key": f"download_{uuid.uuid4()}"
                    }
                    st.download_button(**download_button)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here are the first 10 records. Full results available for download.",
                        "dataframe": df.head(10),
                        "download": download_button,
                        "plot": fig
                    })
                else:
                    st.dataframe(df)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here are the results:",
                        "dataframe": df,
                        "download": None,
                        "plot": fig
                    })

                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

            # Handle text responses (fallback)
            else:
                fig = None
                if plt.get_fignums():
                    fig = plt.gcf()

                placeholder.markdown(output if isinstance(output, str) else str(output))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output if isinstance(output, str) else str(output),
                    "plot": fig,
                    "dataframe": None,
                    "download": None
                })

                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

        except Exception as e:
            placeholder.error(f"Agent error: {e}")