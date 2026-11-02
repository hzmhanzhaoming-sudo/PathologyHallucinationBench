"""API-only entry point. Credentials are environment variables, never source code."""
from launcher import main
if __name__ == '__main__': main(api=True)
