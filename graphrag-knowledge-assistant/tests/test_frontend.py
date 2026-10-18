"""
Stage 6 Frontend & Full-Stack Integration Test
Verifies that FastAPI serves:
1. /health
2. Static dashboard at /
3. CSS at /css/styles.css
4. JS at /js/api.js, /js/graph-viz.js, /js/app.js
5. Core API endpoints: /documents, /graph/stats, /graph/nodes, /query
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app

def run_all_tests():
    print("Testing Stage 6 Frontend Integration...", flush=True)
    with TestClient(app) as client:
        # 1. Health
        resp = client.get("/health")
        assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        print("[PASS] /health: OK", flush=True)

        # 2. Static HTML
        resp = client.get("/")
        assert resp.status_code == 200, f"Root static page failed: {resp.status_code}"
        assert "GraphRAG Knowledge Intelligence" in resp.text
        print("[PASS] GET / (index.html): OK", flush=True)

        # 3. Static Assets
        for asset in ["/css/styles.css", "/js/api.js", "/js/graph-viz.js", "/js/app.js"]:
            resp = client.get(asset)
            assert resp.status_code == 200, f"Asset {asset} failed with {resp.status_code}"
            assert len(resp.text) > 50, f"Asset {asset} is unexpectedly empty"
            print(f"[PASS] GET {asset}: OK ({len(resp.text)} bytes)", flush=True)

        # 4. Documents
        doc_resp = client.get("/documents")
        assert doc_resp.status_code == 200
        docs = doc_resp.json()
        print(f"[PASS] GET /documents: OK ({docs['total']} documents found)", flush=True)

        # 5. Graph stats
        graph_resp = client.get("/graph/stats")
        assert graph_resp.status_code == 200
        gstats = graph_resp.json()
        print(f"[PASS] GET /graph/stats: OK ({gstats['num_nodes']} nodes, {gstats['num_edges']} edges)", flush=True)

        # 6. Graph nodes & edges
        nodes_resp = client.get("/graph/nodes")
        assert nodes_resp.status_code == 200
        edges_resp = client.get("/graph/edges")
        assert edges_resp.status_code == 200
        print(f"[PASS] GET /graph/nodes & /graph/edges: OK ({len(nodes_resp.json())} nodes, {len(edges_resp.json())} edges)", flush=True)

        # 7. Query endpoint with hybrid mode
        print("Running Hybrid Query test (vector + graph RRF)...", flush=True)
        query_resp = client.post(
            "/query",
            json={
                "query": "What is GraphRAG?",
                "mode": "hybrid",
                "top_k_chunks": 2,
                "max_graph_hops": 1
            }
        )
        assert query_resp.status_code == 200, f"Query failed: {query_resp.text}"
        qdata = query_resp.json()
        assert "answer" in qdata
        print(f"[PASS] POST /query (hybrid mode): OK (Generated answer: {len(qdata['answer'])} chars, {len(qdata['sources'])} sources)", flush=True)

        # 8. Test Greeting
        hello_resp = client.post("/query", json={"query": "hello", "mode": "hybrid"})
        assert hello_resp.status_code == 200
        assert "GraphRAG Knowledge Intelligence Assistant" in hello_resp.json()["answer"]
        print("[PASS] POST /query (greeting 'hello'): OK", flush=True)

        # 9. Test Resume QA: "what is my name"
        name_resp = client.post("/query", json={"query": "what is my name", "mode": "hybrid"})
        assert name_resp.status_code == 200
        assert "Tarun R" in name_resp.json()["answer"]
        print(f"[PASS] POST /query ('what is my name'): OK -> Extracted '{name_resp.json()['answer'].splitlines()[0]}'", flush=True)

    print("\n[SUCCESS] ALL STAGE 6 INTEGRATION TESTS PASSED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    run_all_tests()
