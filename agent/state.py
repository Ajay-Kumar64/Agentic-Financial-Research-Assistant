from typing import TypedDict

class AgentState(TypedDict):

    current_query:str

    plan:str

    tool_calls_count:int

    tools_used:list

    last_tool_output:dict

    confidence_score:float

    task_complete:bool