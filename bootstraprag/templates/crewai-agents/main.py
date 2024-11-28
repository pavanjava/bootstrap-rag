from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="ollama/llama3.2:latest", base_url="http://localhost:11434", temperature=0.8, timeout=300)

# Define the Prompt Supervisor agent
prompt_supervisor = Agent(
    role='Prompt Supervisor',
    goal='Ensure all agent prompts are clear, effective, and aligned with users objectives.',
    backstory=(
        "As a Prompt Supervisor, you have a keen eye for detail and a deep understanding "
        "of effective communication strategies. Your mission is to review and refine prompts "
        "to maximize the performance of AI agents."
    ),
    llm=llm
)

# Define a task for the Prompt Supervisor to review and enhance the Senior Researcher's prompt
prompt_supervisor_task = Task(
    description=(
        "Review the prompt provided to the {topic} Senior prompt supervisor, assessing its clarity, "
        "effectiveness, and alignment with the project's objectives. Provide constructive feedback "
        "and suggest improvements to enhance the agent's performance."
    ),
    expected_output=(
        "A detailed evaluation of the original prompt, including specific suggestions for improvement "
        "and a revised version of the prompt that optimizes clarity and effectiveness along with few shot of examples."
    ),
    agent=prompt_supervisor
)


def main():
    # Forming the crew and kicking off the process
    crew = Crew(
        agents=[prompt_supervisor],
        tasks=[prompt_supervisor_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'Get Financial data for 2023'})
    print(result)


if __name__ == "__main__":
    main()
