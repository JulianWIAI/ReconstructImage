"""
main.py — entry point.
All application logic lives in SBS/dashboard_controller.py.
"""

from SBS.dashboard_controller import ShowcaseDashboard


def main() -> None:
    app = ShowcaseDashboard()
    app.run()


if __name__ == "__main__":
    main()
