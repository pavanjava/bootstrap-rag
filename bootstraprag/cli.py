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
@click.option('--framework', type=click.Choice(['llamaindex', 'langchain', 'haystack', 'None']),
              prompt="Which technology would you like to use?")
@click.option('--template', type=click.Choice([]), prompt=False)
@click.option('--observability', type=click.Choice([]), prompt=False)
def create(project_name, framework, template, observability):
    template_choices = []
    observability_choices = []

    if framework == 'llamaindex' or framework == 'langchain' or framework == 'haystack':
        template_choices = ['simple-rag', 'self-rag', 'rag-with-cot', 'rag-with-ReACT', 'rag-with-HyDE']
    elif framework == 'None':
        template_choices = ['simple-search', 'hybrid-search']

    template = click.prompt("Which template would you like to use?",
                            type=click.Choice(template_choices)
                            )
    if framework == 'llamaindex' or framework == 'langchain' or framework == 'haystack':
        observability_choices = ['Yes', 'No']
        observability = click.prompt("Do you wish to enable observability?",
                                     type=click.Choice(observability_choices)
                                     )
    click.echo(f'You have selected framework: {framework} and template: {template} and observability: {observability}')


cli.add_command(create)

if __name__ == "__main__":
    cli()
