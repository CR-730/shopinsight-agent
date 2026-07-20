from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PACKAGE = "business_" + "binding"
LEGACY_PACKAGE_DIR = ROOT / f"app/agent/{LEGACY_PACKAGE}"
LEGACY_FILES = [
    ROOT / f"app/agent/nodes/{LEGACY_PACKAGE}.py",
]


def test_graph_never_imports_legacy_package():
    source = (ROOT / "app/agent/graph.py").read_text(encoding="utf-8-sig")

    assert f"app.agent.{LEGACY_PACKAGE}" not in source
    assert f"nodes.{LEGACY_PACKAGE}" not in source


def test_legacy_files_are_deleted():
    for path in LEGACY_FILES:
        assert not path.exists(), path
    assert not list(LEGACY_PACKAGE_DIR.glob("*.py"))


def test_node_writes_only_the_semantic_plan_contract():
    source = (ROOT / "app/agent/nodes/semantic_planning.py").read_text(
        encoding="utf-8"
    )
    assert "build_semantic_plan" in source
    assert ("validate_" + LEGACY_PACKAGE + "_state") not in source


def test_readme_documents_only_the_canonical_node():
    readme = (ROOT / "README.md").read_text(encoding="utf-8-sig")

    assert "上下文构建 → 语义规划 → 上下文压缩" in readme
    assert "semantic_planning/" in readme
    assert LEGACY_PACKAGE not in readme
