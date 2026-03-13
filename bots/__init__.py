from . import email_report

BOTS = {
    "email_report": {
        "label": "Email Report",
        "runner": email_report.run,
        "runner_stream": email_report.run_stream,
    },
}
