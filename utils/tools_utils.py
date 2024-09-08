import platform
import subprocess
import re


def get_ip_address():
    current_os = platform.system()  # Detect the OS (Windows, Linux, Darwin for macOS)

    if current_os == "Windows":
        # For Windows, using ipconfig
        result = subprocess.run(["ipconfig"], capture_output=True, text=True)
        return parse_ip_address(result.stdout)

    elif current_os == "Linux" or current_os == "Darwin":
        # For Linux or macOS, using ip or ifconfig
        try:
            result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
        except FileNotFoundError:
            result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        return parse_ip_address(result.stdout)

    else:
        return "Unsupported OS"


def parse_ip_address(output):
    # Look for lines containing 'inet' (for IPv4) or 'inet6' (for IPv6)
    ipv4_pattern = re.compile(r'inet (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
    ipv6_pattern = re.compile(r'inet6 ([a-f0-9:]+)')

    # Find all matches for both IPv4 and IPv6
    ipv4_addresses = ipv4_pattern.findall(output)
    ipv6_addresses = ipv6_pattern.findall(output)

    # Filter out loopback addresses
    ipv4_addresses = [ip for ip in ipv4_addresses if not ip.startswith("127.")]
    ipv6_addresses = [ip for ip in ipv6_addresses if not ip.startswith("::1")]

    if ipv4_addresses:
        return f"IPv4 Address: {ipv4_addresses[0]}"
    elif ipv6_addresses:
        return f"IPv6 Address: {ipv6_addresses[0]}"
    else:
        return "No IP address found."


# Run the function to test
# print(get_ip_address())
