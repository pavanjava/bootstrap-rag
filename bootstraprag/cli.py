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
@click.option('--technology', type=click.Choice(['llamaindex', 'langchain', 'haystack']),
              prompt="Which technology would you like to use?")
@click.option('--template', type=click.Choice([]), prompt=False)
def create(project_name, technology, template):
    if technology == 'llamaindex' or technology == 'langchain' or technology == 'haystack':
        template_choices = ['simple-rag', 'self-rag', 'rag-with-cot', 'rag-with-ReACT', 'rag-with-HyDE']
    elif technology == 'qdrant':
        template_choices = ['simple-rag', 'self-rag', 'rag-with-cot', 'rag-with-ReACT', 'rag-with-HyDE']

    template = click.prompt("Which template would you like to use?",
                            type=click.Choice(template_choices)
                            )

    click.echo(f'You have selected technology: {technology} and template: {template}')


cli.add_command(create)

if __name__ == "__main__":
    cli()
