"""
Unit tests for Call Graph & AST Dependency Topology Tool in codeAtlas.
"""
from pathlib import Path
import tempfile
import networkx as nx

from agent.tools.graph_tool import (
    build_dependency_graph,
    analyze_impact,
    query_callers,
    get_topology_summary,
    _extract_calls_from_text,
    _extract_imports_from_source,
)
from agent.graph import invoke


def test_extract_calls_from_text():
    code = """
def process_data(items):
    cleaned = sanitize_input(items)
    score = calculate_average(cleaned)
    if is_valid(score):
        save_result(score)
    return score
"""
    calls = _extract_calls_from_text(code)
    assert "sanitize_input" in calls
    assert "calculate_average" in calls
    assert "is_valid" in calls
    assert "save_result" in calls
    # Keywords must be filtered out
    assert "if" not in calls
    assert "return" not in calls


def test_extract_imports_from_source():
    py_code = """
import os
import sys
from utils.math import calculate_average, sqrt
from config import DATA_DIR
"""
    imports = _extract_imports_from_source(py_code, ".py")
    assert "os" in imports
    assert "sys" in imports
    assert "utils" in imports
    assert "config" in imports

    js_code = """
import React, { useState } from 'react';
import { ChatBox } from './components/ChatBox';
const api = require('./services/api');
"""
    js_imports = _extract_imports_from_source(js_code, ".jsx")
    assert "ChatBox" in js_imports
    assert "api" in js_imports


def test_build_dependency_graph_and_impact_analysis(tmp_path):
    chunks = [
        {
            "id": "math_utils.py::calculate_average@10",
            "file": "math_utils.py",
            "name": "calculate_average",
            "start_line": 10,
            "end_line": 20,
            "code": "def calculate_average(nums):\n    return sum(nums) / len(nums)",
            "type": "function_definition",
            "language": "python",
        },
        {
            "id": "analyzer.py::analyze_scores@15",
            "file": "analyzer.py",
            "name": "analyze_scores",
            "start_line": 15,
            "end_line": 30,
            "code": "def analyze_scores(data):\n    avg = calculate_average(data)\n    return avg",
            "type": "function_definition",
            "language": "python",
        },
        {
            "id": "reporter.py::generate_report@5",
            "file": "reporter.py",
            "name": "generate_report",
            "start_line": 5,
            "end_line": 25,
            "code": "def generate_report(students):\n    res = analyze_scores(students)\n    print(res)",
            "type": "function_definition",
            "language": "python",
        },
    ]

    G = build_dependency_graph(chunks, repo_dir=tmp_path)
    assert G.number_of_nodes() >= 3
    assert G.has_edge("analyzer.py::analyze_scores@15", "math_utils.py::calculate_average@10")
    assert G.has_edge("reporter.py::generate_report@5", "analyzer.py::analyze_scores@15")

    # Impact analysis on calculate_average:
    # Direct caller: analyze_scores
    # Transitive caller: generate_report
    impact = analyze_impact("calculate_average", G=G)
    assert impact["status"] == "ok"
    assert impact["direct_caller_count"] == 1
    assert impact["direct_callers"][0]["name"] == "analyze_scores"
    assert impact["transitive_caller_count"] == 1
    assert impact["transitive_callers"][0]["name"] == "generate_report"
    assert "HIGH" in impact["risk_level"] or "MEDIUM" in impact["risk_level"]

    # Topology summary
    summary = get_topology_summary(G)
    assert summary["total_calls"] == 2
    assert len(summary["top_called_symbols"]) >= 1


def test_agent_graph_intent_routing():
    # Test LangGraph query routing to graph_node
    response = invoke("which functions break if I change calculate_average?")
    assert "Impact" in response or "Blast Radius" in response or "Symbol" in response

    topology_response = invoke("show call graph summary")
    assert "Codebase AST Call Graph" in topology_response


if __name__ == "__main__":
    print("Testing call extraction...")
    test_extract_calls_from_text()
    print("Testing import extraction...")
    test_extract_imports_from_source()
    print("Testing graph construction and impact analysis...")
    with tempfile.TemporaryDirectory() as tmp:
        test_build_dependency_graph_and_impact_analysis(Path(tmp))
    print("Testing agent graph routing...")
    test_agent_graph_intent_routing()
    print("ALL TESTS PASSED SUCCESSFULLY!")
