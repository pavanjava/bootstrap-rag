import click
import shutil
from pathlib import Path
import os
import zipfile


@click.group()
def cli():
    pass

# used for downloading the project as zip.
def create_zip(project_name):
    zip_path = shutil.make_archive(project_name, 'zip', project_name)
    return zip_path


@click.command()
@click.argument('project_name')
@click.option('--framework', type=click.Choice(['llamaindex']),
              prompt="Which technology would you like to use (leave blank, if you want to use qdrant direct search)?",
              default='', required=False)
@click.option('--template', type=click.Choice([]), prompt=False)
@click.option('--observability', type=click.Choice([]), prompt=False)
def create(project_name, framework, template, observability):
    template_choices = []
    observability_choices = []

    if framework == 'llamaindex' or framework == 'langchain' or framework == 'haystack':
        template_choices = ['simple-rag', 'rag-with-react', 'rag-with-hyde']
    elif framework == '':
        framework = 'qdrant'
        template_choices = ['simple-search']

    template = click.prompt("Which template would you like to use?",
                            type=click.Choice(template_choices)
                            )
    if framework == 'llamaindex' or framework == 'langchain' or framework == 'haystack':
        observability_choices = ['Yes', 'No']
        observability = click.prompt("Do you wish to enable observability?",
                                     type=click.Choice(observability_choices)
                                     )

    click.echo(f'You have selected framework: {framework} and template: {template} and observability: {observability}')
    download_and_extract_template(project_name, framework, template, observability)


def download_and_extract_template(project_name, framework, template, observability_selection):
    if observability_selection == 'Yes':
        folder_name = template.replace('-', '_') + '_observability'
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