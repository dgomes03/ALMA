"""
Tool definitions for the RAG Framework.
Centralizes all tool schemas to avoid duplication. Descriptions come from
core/prompts.py so they follow PROMPT_LANGUAGE.
"""

from core import prompts
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
                    "description": prompts.get("fn_search_documents"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": prompts.get("fn_search_documents_query")
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
                    "description": prompts.get("fn_retrieve_document_by_name"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": prompts.get("fn_retrieve_document_by_name_document_name")
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
                    "description": prompts.get("fn_list_available_documents"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_keyword": {
                                "type": "string",
                                "description": prompts.get("fn_list_available_documents_filter_keyword")
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": prompts.get("fn_search_wikipedia"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": prompts.get("fn_search_wikipedia_query")}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "duckduckgo_search",
                    "description": prompts.get("fn_duckduckgo_search"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": prompts.get("fn_duckduckgo_search_query")}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_url_content",
                    "description": prompts.get("fn_fetch_url_content"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": prompts.get("fn_fetch_url_content_url")},
                            "max_chars": {"type": "integer", "description": prompts.get("fn_fetch_url_content_max_chars"), "default": 5000}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": prompts.get("fn_query_database"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": prompts.get("fn_query_database_db_name")
                            },
                            "sql_query": {
                                "type": "string",
                                "description": prompts.get("fn_query_database_sql_query")
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
                    "description": prompts.get("fn_list_databases"),
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
                    "description": prompts.get("fn_get_database_schema"),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "db_name": {
                                "type": "string",
                                "description": prompts.get("fn_get_database_schema_db_name")
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
                "description": prompts.get("fn_activate_advanced_research"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": prompts.get("fn_activate_advanced_research_query")
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    return tools
