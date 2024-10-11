from dotenv import load_dotenv, find_dotenv
from phoenix.evals import HallucinationEvaluator, QAEvaluator, run_evals, OpenAIModel
import pandas as pd

load_dotenv(find_dotenv())

df = pd.read_csv('input_data.csv')

# Set your OpenAI API key
eval_model = OpenAIModel(model="gpt-4o")

# Define your evaluators
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_evaluator = QAEvaluator(eval_model)

# We have to make some minor changes to our dataframe to use the column names expected by our evaluators
# for `hallucination_evaluator` the input df needs to have columns 'output', 'input', 'context'
# for `qa_evaluator` the input df needs to have columns 'output', 'input', 'reference'
df["context"] = df["reference"]
df.rename(columns={"query": "input", "response": "output"}, inplace=True)
assert all(column in df.columns for column in ["output", "input", "context", "reference"])

# Run the evaluators, each evaluator will return a dataframe with evaluation results
# We upload the evaluation results to Phoenix in the next step
hallucination_eval_df, qa_eval_df = run_evals(
    dataframe=df, evaluators=[hallucination_evaluator, qa_evaluator], provide_explanation=True
)

results_df = df.copy()
results_df["hallucination_eval"] = hallucination_eval_df["label"]
results_df["hallucination_explanation"] = hallucination_eval_df["explanation"]
results_df["qa_eval"] = qa_eval_df["label"]
results_df["qa_explanation"] = qa_eval_df["explanation"]

results_df.to_csv('evaluation_report.csv')
