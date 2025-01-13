from langgraph.graph import END, StateGraph
import asyncio
from agents import company_researcher, use_case_generator, resource_collector, InnoVisor

workflow = StateGraph(InnoVisor)

workflow.add_node("Research Agent", company_researcher)
workflow.add_node("Use Case Generator Agent", use_case_generator)
workflow.add_node("Resource Collector Agent", resource_collector)

workflow.set_entry_point("Research Agent")
workflow.set_finish_point("Resource Collector Agent")

workflow.add_edge("Research Agent","Use Case Generator Agent")
workflow.add_edge("Use Case Generator Agent","Resource Collector Agent")

app = workflow.compile()

inputs = {"company_name": "Myntra", "company_url": "https://www.myntra.com/"}
config = {"recursion_limit": 10}


async def run_with_aggregation():
    aggregated_result = {}
    async for event in app.astream(inputs, config=config):
        aggregated_result.update(event)
        print(f"Intermediate Event: {event}")
    print("\nFinal Aggregated Result:", aggregated_result)


asyncio.run(run_with_aggregation())