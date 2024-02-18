from langchain_iris import IRISVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_together import Together
from dotenv import load_dotenv

class RAG():
    def __init__(self, company, model):
        embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5")

        username = 'SUPERUSER'
        password = 'oasis' 
        hostname = 'localhost' 
        port = '1972' 
        namespace = 'USER'
        CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
        COLLECTION_NAME = "vectordb"

        self.db = IRISVector(
            embedding_function=embeddings,
            dimension=768,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING)

        self.retriever = self.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={
            "k": 50,
            "score_threshold": .15,
            "filter": {}
        })

        self.model = model

        self.company = company

        template = """
        <s>[INST] You are an agent speaking with a representative from {company}.
        Your goal is to assist the representative with questions regarding their sustainability practices.
        You are provided context regarding {company}'s sustainablity practices and your goal is to compare
        their practices with practices of other companies, for which context is given to you, in order to
        help them improve.
        Be critical with your response as well by noticing when metrics are missing or where they could be better.
        Answer the question with a detailed response when necessary, based only on the following context, incorporating data from other companies:
        {context}

        Conversation History:
        {history}

        Representative: {question} [/INST] 
        """
        self.prompt = ChatPromptTemplate.from_template(template)

        # Initialize conversation history
        self.conversation_history = ""
    
    def update_history(self, question, answer):
        self.conversation_history += f"Representative: {question}\nAgent: {answer}\n"

    def get_response(self, input_query):
        # Define the chain with updated conversation history in the context
        chain = (
            {
                "context": self.retriever, 
                "history": lambda x: self.conversation_history, 
                "company": lambda x: self.company, 
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        
        output = chain.invoke(input_query)
        self.update_history(input_query, output)
        return output

    def chat_interface(self):
        while True:
            user = input(">>> ")
            if user == "stop":
                break
            print(self.get_response(user))

if __name__ == "__main__":
    company = "NVIDIA Corporation"
    load_dotenv()
    model = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1024,
        top_k=50,
    )
    r = RAG(company, model)
    r.chat_interface()
