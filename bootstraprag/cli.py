import click
import shutil
from pathlib import Path
import os
import zipfile


@click.group()
def cli():
    pass


def create_zip(project_name):
    zip_path = shutil.make_archive(project_name, 'zip', project_name)
    return zip_path


@click.command()
@click.argument('project_name')
@click.option('--template', type=click.Choice(['simple-search', 'hybrid-search', 'llamaindex-rag', 'rag-with-cot', 'rag-with-ReACT', 'rag-with-HyDE']),
              prompt="Which template would you like to use?")
@click.option('--framework', type=click.Choice(['llamaindex', 'langchain', 'haystack']),
              prompt="Which framework would you like to use?")
@click.option('--observability', type=click.Choice(['Yes', 'No']), prompt="Would you like to set up observability?")
@click.option('--api_key', prompt="Please provide your OpenAI API key (leave blank to skip)", default='',
              required=False)
@click.option('--data_source', type=click.Choice(['PDF', 'TXT']),
              prompt="Which data source would you like to use?")
@click.option('--additional_data_source', type=click.Choice(['Yes', 'No']),
              prompt="Would you like to add another data source?")
@click.option('--vector_db', type=click.Choice(['Yes', 'No']), prompt="Would you like to use a vector database?")
def create(project_name, template, framework, observability, api_key, data_source, additional_data_source, vector_db):
    """Creates a new project with the specified type."""
    template_path = Path(__file__).parent / 'templates' / template
    project_path = Path.cwd() / project_name

    if project_path.exists():
        click.echo(f"Error: Project directory {project_name} already exists!")
        return

    shutil.copytree(template_path, project_path)

    # Optionally handle the API key and other parameters here if necessary

    zip_path = create_zip(project_name)

    click.echo(f"Created {template} project at {project_path}")
    click.echo(f"Project archived as {zip_path}")


cli.add_command(create)

if __name__ == "__main__":
    cli()
