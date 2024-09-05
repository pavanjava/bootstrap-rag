import pytest
from click.testing import CliRunner
from unittest.mock import patch
from bootstraprag.cli import cli, download_and_extract_template  # replace 'your_cli_module' with the actual module name


@pytest.fixture
def runner():
    return CliRunner()


def test_create_project_without_overwrite(runner):
    with patch('your_cli_module.Path.exists', return_value=True):
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert "Error: Project directory test_project already exists!" in result.output


def test_create_project_success(runner):
    with patch('your_cli_module.Path.exists', return_value=False), \
            patch('your_cli_module.shutil.copytree') as mock_copy:
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert "Project test_project created successfully" in result.output
        mock_copy.assert_called_once()


def test_framework_selection_llamaindex(runner):
    with patch('InquirerPy.inquirer.select.execute', return_value='llamaindex'):
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert 'You have selected framework: llamaindex' in result.output


def test_template_selection_simple_rag(runner):
    with patch('InquirerPy.inquirer.select.execute', side_effect=['llamaindex', 'simple-rag']):
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert 'You have selected template: simple-rag' in result.output


def test_observability_selection(runner):
    with patch('InquirerPy.inquirer.select.execute', side_effect=['llamaindex', 'simple-rag', 'Yes']):
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert 'You have selected observability: Yes' in result.output


def test_download_and_extract_template_with_observability():
    with patch('your_cli_module.shutil.copytree') as mock_copy:
        download_and_extract_template('test_project', 'llamaindex', 'simple-rag', 'Yes')
        mock_copy.assert_called_once()


def test_download_and_extract_template_without_observability():
    with patch('your_cli_module.shutil.copytree') as mock_copy:
        download_and_extract_template('test_project', 'llamaindex', 'simple-rag', 'No')
        mock_copy.assert_called_once()
