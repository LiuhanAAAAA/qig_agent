from rich.console import Console
console = Console()

def log_info(msg: str):
    console.print(f"[bold cyan][INFO][/bold cyan] {msg}")

def log_warn(msg: str):
    console.print(f"[bold yellow][WARN][/bold yellow] {msg}")

def log_error(msg: str):
    console.print(f"[bold red][ERR][/bold red] {msg}")
