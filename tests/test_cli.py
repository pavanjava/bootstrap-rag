import pytest
from click.testing import CliRunner
from unittest.mock import patch
from bootstraprag.cli import cli, download_and_extract_template  # replace 'your_cli_module' with the actual module name


@pytest.fixture
def runner():
    return CliRunner()


def test_create_project_without_overwrite(runner):
    # Mock Path.exists to simulate that the project directory already exists
    with patch('pathlib.Path.exists', return_value=True), \
            patch('shutil.copytree') as mock_copy, \
            patch('InquirerPy.inquirer.select') as mock_inquirer_select:
        # Ensure copytree wasn't called since the directory exists
        mock_copy.assert_not_called()
        # Ensure no prompt was triggered
        mock_inquirer_select.assert_not_called()

        # The prompts should not be triggered because the function should exit early
        result = runner.invoke(cli, ['create', 'test_project'])

        # Assert that the command exited with success (in this case we expect no project creation)
        assert result.exit_code == 0, f"Exit code was {result.exit_code}, output: {result.output}"
        assert "Error: Project directory test_project already exists!" in result.output


def test_framework_selection_llamaindex(runner):
    with patch('InquirerPy.inquirer.select.execute', return_value='llamaindex'):
        result = runner.invoke(cli, ['create', 'test_project'])
        assert result.exit_code == 0
        assert 'You have selected framework: llamaindex' in result.output


def test_download_and_extract_template_with_observability():
    with patch('shutil.copytree') as mock_copy:
        download_and_extract_template('test_project', 'llamaindex', 'simple-rag', 'Yes')
        mock_copy.assert_called_once()


def test_download_and_extract_template_without_observability():
    with patch('shutil.copytree') as mock_copy:
        download_and_extract_template('test_project', 'llamaindex', 'simple-rag', 'No')
        mock_copy.assert_called_once()
