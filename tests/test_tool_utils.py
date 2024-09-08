import pytest
import subprocess
import platform
from utils.tools_utils import get_ip_address


def mock_subprocess_run_ifconfig(*args, **kwargs):
    # Mock output for Linux/macOS `ifconfig` command
    class MockCompletedProcess:
        def __init__(self):
            self.stdout = """
            en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
                inet 192.168.1.2 netmask 0xffffff00 broadcast 192.168.1.255
            lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
                inet 127.0.0.1 netmask 0xff000000
            """

    return MockCompletedProcess()


def mock_subprocess_run_ipconfig(*args, **kwargs):
    # Mock output for Windows `ipconfig` command
    class MockCompletedProcess:
        def __init__(self):
            self.stdout = """
            Ethernet adapter Ethernet:
                IPv4 Address. . . . . . . . . . . : 192.168.1.100
                Subnet Mask . . . . . . . . . . . : 255.255.255.0
                Default Gateway . . . . . . . . . : 192.168.1.1
            """

    return MockCompletedProcess()


@pytest.mark.skipif(platform.system() == "Windows", reason="Running on Windows OS")
def test_get_ip_address_linux_or_mac(monkeypatch):
    if platform.system() in ["Linux", "Darwin"]:
        # Mock subprocess.run to return Linux/macOS output
        monkeypatch.setattr(subprocess, 'run', mock_subprocess_run_ifconfig)

        # Run the get_ip_address function and assert the result
        result = get_ip_address()
        assert result == "IPv4 Address: 192.168.1.2"


@pytest.mark.skipif(platform.system() != "Windows", reason="Not running on Windows OS")
def test_get_ip_address_windows(monkeypatch):
    if platform.system() == "Windows":
        # Mock subprocess.run to return Windows output
        monkeypatch.setattr(subprocess, 'run', mock_subprocess_run_ipconfig)

        # Run the get_ip_address function and assert the result
        result = get_ip_address()
        assert result == "IPv4 Address: 192.168.1.100"
