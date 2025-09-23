import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.chat_models import init_chat_model
from rag import RAGPipeline

class MeetingAgent:
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        # Initialize Gemini instead of OpenAI
        self.model = init_chat_model(
            "gemini-1.5-pro",           # You can also try "gemini-1.5-flash" for faster responses
            model_provider="google",
            api_key=os.getenv("AIzaSyBhnL0ISv5EFUv8A3XKRuxlbsnAPR5NlkM")  # Make sure GEMINI_API_KEY is set
        )
        self.tools = [
            Tool(
                name="DocumentRetriever",
                func=self.rag.retrieve,
                description="Retrieve relevant information from uploaded documents and meeting transcript."
            )
        ]
        self.agent = create_react_agent(self.model, self.tools)

    def process_question(self, question):
        return self.agent.run(f"Given the meeting context and documents, answer: {question}")

if __name__ == "__main__":
    rag = RAGPipeline()
    rag.index_documents(["sample_contract.pdf"])
    agent = MeetingAgent(rag)
    answer = agent.process_question("What is the penalty for late delivery?")
    print(f"Answer: {answer}")
