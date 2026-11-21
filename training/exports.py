"""Portable training-history exports, independent of the UI."""
import csv
import io


def history_csv(history):
    """Include the ordered union of metrics, even when later rows add fields."""
    records = list(history)
    if not records:
        return ''
    output = io.StringIO(newline='')
    fields = list(dict.fromkeys(key for record in records for key in record))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()
