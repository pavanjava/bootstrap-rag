## Phoenix Evaluations

This repository provides a script for evaluating model-generated responses using Phoenix's `HallucinationEvaluator` and `QAEvaluator`.
To start evaluating run `bootstraprag create phoenix_evals`
select the specific template shown on cli

```text
? Which technology would you like to use? standalone-evaluations
? Which template would you like to use? 
  deep-evals
  mlflow-evals
❯ phoenix-evals
  ragas-evals
```

just replace the `input_data.csv` with your own data, the file has following columns
`id,reference,query,response`.

### How to execute?
- run `python basic_evaluations.py`

### What to expect?
- At the end of process, you can see the `evaluation_report.csv` is created and kept in the parent folder where you can see different aspects of evaluations carried on your input data.