import os
import json
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.generate import ask_question
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def eval_dataset():
    with open("eval/eval_data.json", "r") as f:
        data = json.load(f)
    
    questions = []
    ground_truths = []
    answers = []
    contexts = []
    
    for item in data:
        result = ask_question(item["question"])
        
        questions.append(item["question"])
        ground_truths.append(item["ground_truth"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    return dataset

def test_rag_performance(eval_dataset):
    # Configure Ragas to use Groq and HuggingFace instead of default OpenAI
    # Use evaluate directly, but wrap metrics
    groq_llm = ChatGroq(model_name="llama-3.1-8b-instant")
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We must explicitly set the llm and embeddings on each metric
    faithfulness.llm = groq_llm
    answer_relevancy.llm = groq_llm
    answer_relevancy.embeddings = hf_embeddings
    context_precision.llm = groq_llm
    
    result = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    )
    
    print("\n--- Ragas Evaluation Scores ---")
    print(result)
    
    assert result["faithfulness"] >= 0.8, f"Faithfulness too low: {result['faithfulness']}"
    assert result["answer_relevancy"] >= 0.8, f"Answer Relevancy too low: {result['answer_relevancy']}"
    assert result["context_precision"] >= 0.8, f"Context Precision too low: {result['context_precision']}"
