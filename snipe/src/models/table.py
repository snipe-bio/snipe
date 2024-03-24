from dataclasses import dataclass, field
from typing import List, Dict, Any
import threading

@dataclass
class Table:
    columns: List[str]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_row(self, row_data: Dict[str, Any]):
        with self.lock:
            filtered_row = {k: v for k, v in row_data.items() if k in self.columns}
            for column in self.columns:
                if column not in filtered_row:
                    filtered_row[column] = None
            self.rows.append(filtered_row)

    def get_rows(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [row.copy() for row in self.rows]

    def export(self, delimiter: str = ",") -> str:
        """Exports the table data as a string with the specified delimiter."""
        with self.lock:
            header = delimiter.join(self.columns)
            rows_str = "\n".join(delimiter.join(str(row.get(col, '')) for col in self.columns) for row in self.rows)
            return f"{header}\n{rows_str}"

    def __str__(self):
        """Provides a simple string representation of the table for debugging purposes."""
        return self.export(delimiter="\t")
