import click
import shutil
from pathlib import Path
from InquirerPy import inquirer


@click.group()
def cli():
    pass


# used for downloading the project as zip.
def create_zip(project_name):
    zip_path = shutil.make_archive(project_name, 'zip', project_name)
    return zip_path


@click.command()
@click.argument('project_name')
@click.option('--framework', type=click.Choice([]), prompt=False)
@click.option('--template', type=click.Choice([]), prompt=False)
@click.option('--observability', type=click.Choice([]), prompt=False)
def create(project_name, framework, template, observability):
    template_choices = []
    framework_choices = ['llamaindex', 'langchain', 'standalone-qdrant', 'standalone-evaluations']
    framework = inquirer.select(
        message="Which technology would you like to use?",
        choices=framework_choices
    ).execute()
    if framework == 'llamaindex':
        template_choices = [
            'simple-rag',
            'rag-with-react',
            'rag-with-hyde',
            'rag-with-flare',
            'rag-with-self-correction',
            'rag-with-controllable-agents',
            'llama-deploy-with-simplemq',
            'llama-deploy-with-rabbitmq',
            'llama-deploy-with-kafka'
        ]

    elif framework == 'langchain':
        template_choices = [
            'simple-rag'
        ]
    elif framework == 'standalone-qdrant':
        framework = 'qdrant'
        template_choices = ['simple-search', 'multimodal-search', 'hybrid-search', 'hybrid-search-advanced',
                            'retrieval-quality']
    elif framework == 'standalone-evaluations':
        framework = 'evaluations'
        template_choices = ['deep-evals', 'mlflow-evals', 'phoenix-evals', 'ragas-evals']
    # Use InquirerPy to select template with arrow keys
    template = inquirer.select(
        message="Which template would you like to use?",
        choices=template_choices,
    ).execute()
    if framework == 'llamaindex' or framework == 'langchain' or framework == 'haystack':
        observability_choices = ['Yes', 'No']
        # Use InquirerPy to select observability with arrow keys
        observability = inquirer.select(
            message="Do you wish to enable observability?",
            choices=observability_choices,
        ).execute()

    click.echo(f'You have selected framework: {framework} and template: {template} and observability: {observability}')
    download_and_extract_template(project_name, framework, template, observability)


def download_and_extract_template(project_name, framework, template, observability_selection):
    if observability_selection == 'Yes':
        folder_name = template.replace('-', '_') + '_with_observability'
        base_path = Path(__file__).parent / 'templates' / framework / folder_name
    else:
        base_path = Path(__file__).parent / 'templates' / framework / str(template).replace('-', '_')
    project_path = Path.cwd() / project_name

    if project_path.exists():
        click.echo(f"Error: Project directory {project_name} already exists!")
        return

    try:
        shutil.copytree(base_path, project_path)
        click.echo(f"Project {project_name} created successfully at {project_path}")
    except Exception as e:
        click.echo(f"Error: {e}")


cli.add_command(create)

if __name__ == "__main__":
    cli()
