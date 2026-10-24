"""
Tool definitions for the RAG Framework.
Centralizes all tool schemas to avoid duplication.
"""

from core.config import ADVANCED_RESEARCH


def get_tools_for_standard_generator():
    """
    Returns the tools list for the standard generator.

    """
    tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Searches user's documents for information to respond to the user's query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_document_by_name",
                    "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The name of the document to retrieve (e.g., 'annual_report.pdf', 'data.xlsx', 'notes.md')"
                            }
                        },
                        "required": ["document_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_documents",
                    "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_keyword": {
                                "type": "string",
                                "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Searches Wikipedia for encyclopedic information on a given topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string for Wikipedia."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": "Search the web using DuckDuckGo. Returns 7 results with titles, URLs, and snippets. Use for current events or general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_content",
                    "description": "Fetch and extract the main text content from a web page. ONLY use URLs that were returned by a prior duckduckgo_search or search_wikipedia result. NEVER invent, guess, or construct URLs. NEVER fetch YouTube URLs. NEVER fetch PDF URLs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The full URL to fetch"},
                            "max_chars": {"type": "integer", "description": "Max characters to return (default 5000)", "default": 5000}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Execute a read-only SQL query on a configured database. SELECT statements and CTEs (WITH ... SELECT) are allowed; window functions, subqueries and joins are supported.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to query (use list_databases to see available options)"
                            },
                            "sql_query": {
                                "type": "string",
                                "description": "The SQL query to execute. Must be read-only: a SELECT statement, optionally starting with a WITH clause (CTE)."
                            }
                        },
                        "required": ["db_name", "sql_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_databases",
                    "description": "List all SQL databases available for querying.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_database_schema",
                    "description": "Get the schema (tables and columns) of a database. Use this to understand the database structure before writing queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": "The name of the database to get the schema for"
                            }
                        },
                        "required": ["db_name"]
                    }
                }
            }
        ]

    if ADVANCED_RESEARCH:
        tools.append({
            "type": "function",
            "function": {
                "name": "activate_advanced_research",
                "description": (
                    "Activate advanced multi-step research for complex questions that require "
                    "deep analysis, multi-source research, planning, and synthesis. Draws on both "
                    "the local document corpus (search_documents, list_available_documents, "
                    "retrieve_document_by_name) and the open web (duckduckgo_search + fetch_url_content). "
                    "Use for multi-faceted questions, cross-referencing sources, step-by-step reasoning, "
                    "or comprehensive analysis. Do NOT use for simple factual questions answerable directly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The complex question or task to reason about."
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    return tools
