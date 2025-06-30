"""
text_to_sql.py – Natural language to SQL conversion for Unimind native models.
Provides conversion from natural language queries to SQL statements with schema awareness.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class SQLOperation(Enum):
    """Enumeration of SQL operations."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"

@dataclass
class SQLResult:
    """Result of SQL conversion."""
    sql_query: str
    operation: SQLOperation
    confidence: float
    tables: List[str]
    columns: List[str]
    conditions: List[str]
    explanation: str

class TextToSQL:
    """
    Converts natural language queries to SQL statements.
    Provides schema awareness and query optimization capabilities.
    """
    
    def __init__(self):
        """Initialize the TextToSQL converter."""
        self.schema = {}
        self.query_patterns = {
            SQLOperation.SELECT: [
                r"show me|display|get|find|list|retrieve|fetch|what|which|how many",
                r"select|show|get all|find all|list all"
            ],
            SQLOperation.INSERT: [
                r"add|insert|create new|add new|save|store",
                r"insert into|add to|create record"
            ],
            SQLOperation.UPDATE: [
                r"update|modify|change|edit|alter",
                r"update|modify|change the"
            ],
            SQLOperation.DELETE: [
                r"delete|remove|drop|erase|clear",
                r"delete from|remove from|drop"
            ]
        }
        
        self.column_patterns = {
            "name": ["name", "title", "full name", "user name"],
            "id": ["id", "identifier", "primary key", "key"],
            "email": ["email", "e-mail", "email address"],
            "date": ["date", "time", "timestamp", "created", "updated"],
            "status": ["status", "state", "condition"],
            "count": ["count", "number", "how many", "total"],
            "all": ["all", "everything", "complete", "full"]
        }
        
        self.condition_patterns = {
            "equals": ["is", "equals", "equal to", "="],
            "greater_than": ["greater than", "more than", "above", ">"],
            "less_than": ["less than", "below", "under", "<"],
            "contains": ["contains", "has", "includes", "like"],
            "starts_with": ["starts with", "begins with"],
            "ends_with": ["ends with", "finishes with"]
        }
        
    def set_schema(self, schema: Dict[str, List[str]]) -> None:
        """
        Set the database schema for better query generation.
        
        Args:
            schema: Dictionary mapping table names to column lists
        """
        self.schema = schema
        
    def convert_to_sql(self, natural_query: str) -> SQLResult:
        """
        Convert natural language query to SQL.
        
        Args:
            natural_query: Natural language query string
            
        Returns:
            SQLResult containing the converted SQL and metadata
        """
        query_lower = natural_query.lower().strip()
        
        # Determine operation
        operation = self._detect_operation(query_lower)
        
        # Extract tables and columns
        tables = self._extract_tables(query_lower)
        columns = self._extract_columns(query_lower)
        conditions = self._extract_conditions(query_lower)
        
        # Generate SQL query
        sql_query = self._generate_sql(operation, tables, columns, conditions)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query_lower, operation, tables, columns)
        
        # Generate explanation
        explanation = self._generate_explanation(operation, tables, columns, conditions)
        
        return SQLResult(
            sql_query=sql_query,
            operation=operation,
            confidence=confidence,
            tables=tables,
            columns=columns,
            conditions=conditions,
            explanation=explanation
        )
    
    def _detect_operation(self, query: str) -> SQLOperation:
        """Detect the SQL operation from the query."""
        for operation, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return operation
        return SQLOperation.SELECT  # Default to SELECT
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from the query."""
        tables = []
        
        # Look for table names in schema
        if self.schema:
            for table_name in self.schema.keys():
                if table_name.lower() in query:
                    tables.append(table_name)
        
        # If no tables found in schema, try common table patterns
        if not tables:
            common_tables = ["users", "user", "customers", "customer", "orders", "order", "products", "product"]
            for table in common_tables:
                if table in query:
                    tables.append(table)
        
        return tables if tables else ["unknown_table"]
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from the query."""
        columns = []
        
        # Check for specific column patterns
        for column_type, patterns in self.column_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    if column_type == "all":
                        columns = ["*"]
                        break
                    else:
                        columns.append(column_type)
        
        # If no specific columns found, try to extract from schema
        if not columns and self.schema:
            for table_name, table_columns in self.schema.items():
                if table_name.lower() in query:
                    for col in table_columns:
                        if col.lower() in query:
                            columns.append(col)
        
        return columns if columns else ["*"]
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions from the query."""
        conditions = []
        
        # Look for condition patterns
        for condition_type, patterns in self.condition_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    # Extract the condition value
                    match = re.search(f"{pattern}\\s+(\\w+)", query)
                    if match:
                        value = match.group(1)
                        if condition_type == "equals":
                            conditions.append(f"= '{value}'")
                        elif condition_type == "greater_than":
                            conditions.append(f"> '{value}'")
                        elif condition_type == "less_than":
                            conditions.append(f"< '{value}'")
                        elif condition_type == "contains":
                            conditions.append(f"LIKE '%{value}%'")
                        elif condition_type == "starts_with":
                            conditions.append(f"LIKE '{value}%'")
                        elif condition_type == "ends_with":
                            conditions.append(f"LIKE '%{value}'")
        
        return conditions
    
    def _generate_sql(self, operation: SQLOperation, tables: List[str], columns: List[str], conditions: List[str]) -> str:
        """Generate SQL query string."""
        if operation == SQLOperation.SELECT:
            table_name = tables[0] if tables else "table_name"
            column_list = ", ".join(columns) if columns else "*"
            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            return f"SELECT {column_list} FROM {table_name}{where_clause};"
        
        elif operation == SQLOperation.INSERT:
            table_name = tables[0] if tables else "table_name"
            column_list = ", ".join(columns) if columns else "column1, column2"
            return f"INSERT INTO {table_name} ({column_list}) VALUES (value1, value2);"
        
        elif operation == SQLOperation.UPDATE:
            table_name = tables[0] if tables else "table_name"
            set_clause = "column1 = value1, column2 = value2"
            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            return f"UPDATE {table_name} SET {set_clause}{where_clause};"
        
        elif operation == SQLOperation.DELETE:
            table_name = tables[0] if tables else "table_name"
            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            return f"DELETE FROM {table_name}{where_clause};"
        
        return "SELECT * FROM table_name;"
    
    def _calculate_confidence(self, query: str, operation: SQLOperation, tables: List[str], columns: List[str]) -> float:
        """Calculate confidence score for the conversion."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear operation detection
        if operation != SQLOperation.SELECT:
            confidence += 0.2
        
        # Boost confidence for schema-aware table detection
        if self.schema and any(table in self.schema for table in tables):
            confidence += 0.2
        
        # Boost confidence for specific column detection
        if columns and columns != ["*"]:
            confidence += 0.1
        
        # Boost confidence for condition detection
        if any(condition in query for patterns in self.condition_patterns.values() for condition in patterns):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_explanation(self, operation: SQLOperation, tables: List[str], columns: List[str], conditions: List[str]) -> str:
        """Generate human-readable explanation of the SQL conversion."""
        operation_name = operation.value.lower()
        table_list = ", ".join(tables)
        column_list = ", ".join(columns) if columns else "all columns"
        
        explanation = f"Converted to {operation_name} operation on table(s): {table_list}"
        explanation += f", selecting columns: {column_list}"
        
        if conditions:
            explanation += f", with conditions: {' AND '.join(conditions)}"
        
        return explanation
    
    def optimize_query(self, sql_result: SQLResult) -> SQLResult:
        """
        Optimize the generated SQL query.
        
        Args:
            sql_result: Original SQL result
            
        Returns:
            Optimized SQL result
        """
        # Simple optimization: add LIMIT for large result sets
        if sql_result.operation == SQLOperation.SELECT and "*" in sql_result.columns:
            optimized_sql = sql_result.sql_query.replace(";", " LIMIT 100;")
            return SQLResult(
                sql_query=optimized_sql,
                operation=sql_result.operation,
                confidence=sql_result.confidence,
                tables=sql_result.tables,
                columns=sql_result.columns,
                conditions=sql_result.conditions,
                explanation=sql_result.explanation + " (optimized with LIMIT 100)"
            )
        
        return sql_result

# Module-level instance
text_to_sql = TextToSQL()

def convert_to_sql(natural_query: str) -> SQLResult:
    """Convert natural language to SQL using the module-level instance."""
    return text_to_sql.convert_to_sql(natural_query)

def set_schema(schema: Dict[str, List[str]]) -> None:
    """Set schema using the module-level instance."""
    text_to_sql.set_schema(schema)
