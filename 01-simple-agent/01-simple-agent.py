# This example extends the workshop lab from the Bedrock workshop https://catalog.us-east-1.prod.workshops.aws/workshops/a4bdb007-5600-4368-81c5-ff5b4154f518/en-US/90-agents/91-agents-w-langchain
# It uses DuckDuckGo Search, since it does not require an API key.

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMMathChain
from langchain.tools import DuckDuckGoSearchRun

model_parameters = {
    "temperature": 0.0,
    "top_p": 0.5,
    "max_tokens_to_sample": 2000,
}

# Initalize LLM handler
react_agent_llm = Bedrock(
    model_id="anthropic.claude-instant-v1", model_kwargs=model_parameters
)
math_chain_llm = Bedrock(
    model_id="anthropic.claude-instant-v1",
    model_kwargs={"temperature": 0, "stop_sequences": ["```output"]},
)

# Web Search Tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, years, dates, issues, etc. Worth using for general topics. Use precise questions.",
)

# Math tool
llm_math_chain = LLMMathChain.from_llm(llm=math_chain_llm, verbose=True)
llm_math_chain.llm_chain.prompt.template = """Human: Given a question with a math problem, provide only a single line mathematical expression that solves the problem in the following format. Don't solve the expression only create a parsable expression.
```text
${{single line mathematical expression that solves the problem}}
```

Assistant:
 Here is an example response with a single line mathematical expression for solving a math problem:
```text
37593**(1/5)
```

Human: {question}

Assistant:"""

math_tool = Tool.from_function(
    func=llm_math_chain.run,
    name="Calculator",
    description="Useful for when you need to answer questions about math.",
)

# Set up all tools in an array
tools = [search_tool, math_tool]

# Initialize the agent with the tools
react_agent = initialize_agent(
    tools,
    react_agent_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

prompt_template = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do, Also try to follow steps mentioned above
Action: the action to take, should be one of ["Web Search", "Calculator"]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}

Assistant:
{agent_scratchpad}"""

react_agent.agent.llm_chain.prompt.template = prompt_template

question = "What is Knowledge Bases for Amazon Bedrock? Multiply the month (numerical) of the announcement date of the Knowledge Bases feature by 3."

react_agent(question)
