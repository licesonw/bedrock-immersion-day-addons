# Amazon Bedrock Immersion Day Addons - Lab 01: Simple Agent

## Overview
In this lab we will learn about Agents. Certain applications demand an adaptable sequence of calls to language models and various utilities depending on user input. The Agent interface enables such flexibility for these applications. An agent has availability to a range of resources and selects which ones to utilize based on the user input. Agents are capable of using multiple tools and utilizing the output of one tool as the input for the next.

There are two primary categories of agents:

- **Action agents**: At each interval, determine the subsequent action utilizing the outputs of all previous actions.
- **Plan-and-execute agents**: Determine the complete order of actions initially, then implement them all without updating the plan.

## Lab walkthrough
Follow the steps in this walkthrough and copy the code to a new python file in your lab environment. You can also just copy the content from the file [01-simple-agent.py](01-simple-agent.py).

### Using ReAct: Synergizing Reasoning and Acting in Language Models Framework

Large language models can generate both explanations for their reasoning and task-specific responses in an alternating fashion.

Producing reasoning explanations enables the model to infer, monitor, and revise action plans, and even handle unexpected scenarios. The action step allows the model to interface with and obtain information from external sources such as knowledge bases or environments.

The ReAct framework could enable large language models to interact with external tools to obtain additional information that results in more accurate and fact-based responses.

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMMathChain
from langchain.tools import DuckDuckGoSearchRun
```

Load two unique LLM objects with unique model parameters, one for the agent, and the other for the math chain.

A unique LLM with stop sequences for the math chain helps prevent Claude from being overly verbose while running the math chain.

Additionally, we'll adapt the default template to better fit Claude, as the default Langchain templates are not well fit by default, then append the newly constructed tool to our list of tools.

```python
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
```


We now define the web search tool using DuckDuckGo web search as a tool to be provided to the LLM via LangChain.

```python
# Web Search Tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, years, dates, issues, etc. Worth using for general topics. Use precise questions.",
)
```

Now, we create the math tool as well and create the agent object in LangChain that provide the two tools to the LLM at runtime.

```python
# Math tool
llm_math_chain = LLMMathChain.from_llm(llm=math_chain_llm, verbose=True)
llm_math_chain.llm_chain.prompt.template = """Human: Given a question with a math problem, provide only a single line mathematical expression that solves the problem in the following format. Don't solve the expression only create a parsable expression.
```text
${{single line mathematical expression that solves the problem}}```


Assistant:
 Here is an example response with a single line mathematical expression for solving a math problem:
```text
37593**(1/5)```

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
```

Finally, we add a custom prompt template to the agent chain.

```python
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
```

We can now run the agent chain with a complex question and have it solve the question using its tools.

```python
question = "What is Knowledge Bases for Amazon Bedrock? Multiply the month (numerical) of the announcement date of the Knowledge Bases feature by 3."

react_agent(question)
```

## References
This tutorial is based on the [original Amazon Bedrock workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/a4bdb007-5600-4368-81c5-ff5b4154f518/en-US/90-agents/91-agents-w-langchain) for agents.