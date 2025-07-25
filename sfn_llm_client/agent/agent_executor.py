# from langchain.agents import AgentExecutor, create_openai_tools_agent
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.callbacks import get_openai_callback



# def call_agent(llm_client, tools, configuration):
#     raw_prompt = configuration["system_prompt"]
#     user_query = configuration["user_query"]
#     chat_history = configuration.get("chat_history", None)

#     # clean prompt
#     _prompt = raw_prompt.rstrip("\n").splitlines()
#     _prompt= " ".join(_prompt).replace("'", " ").replace('"', " ")
#     print(_prompt)
#     # Create prompt template with explicit input variables
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", _prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{{query}}"),  # Use query instead of input
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ])

#     agent = create_openai_tools_agent(llm=llm_client, tools=tools, prompt=prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=False, max_iterations=5)
#     # Run the agent
#     # TODO: for cortex?
#     with get_openai_callback() as cb:
#         response = agent_executor.invoke({
#             "input": user_query,
#             "chat_history": chat_history
#         })
#     token_cost_summary = {
#             "prompt_tokens": cb.prompt_tokens,
#             "completion_tokens": cb.completion_tokens,
#             "total_tokens": cb.total_tokens,
#             "total_cost_usd": cb.total_cost,
#         }
#     return response, token_cost_summary