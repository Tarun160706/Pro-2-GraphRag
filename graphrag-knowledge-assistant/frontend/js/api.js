/**
 * GraphRAG Knowledge Assistant — API Client
 * Encapsulates all backend REST API endpoints with robust error handling.
 */

const API_BASE = ''; // Same origin when served via FastAPI StaticFiles

class GraphRagApi {
    constructor(baseUrl = API_BASE) {
        this.baseUrl = baseUrl;
    }

    /**
     * Helper for standardized fetch requests
     */
    async _fetch(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultHeaders = {
            'Accept': 'application/json',
        };

        if (options.body && !(options.body instanceof FormData)) {
            defaultHeaders['Content-Type'] = 'application/json';
        }

        const config = {
            ...options,
            headers: {
                ...defaultHeaders,
                ...(options.headers || {})
            }
        };

        try {
            const response = await fetch(url, config);
            const contentType = response.headers.get('content-type') || '';
            const isJson = contentType.includes('application/json');
            const data = isJson ? await response.json() : await response.text();

            if (!response.ok) {
                const errorMessage = (data && data.detail) 
                    ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
                    : `HTTP Error ${response.status}: ${response.statusText}`;
                throw new Error(errorMessage);
            }

            return data;
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    // ── System Health ──────────────────────────────────────────
    async getHealth() {
        return this._fetch('/health');
    }

    // ── Documents ──────────────────────────────────────────────
    async getDocuments() {
        return this._fetch('/documents');
    }

    async getDocument(documentId) {
        return this._fetch(`/documents/${documentId}`);
    }

    async getDocumentChunks(documentId) {
        return this._fetch(`/documents/${documentId}/chunks`);
    }

    async uploadDocument(file, onProgress) {
        const formData = new FormData();
        formData.append('file', file);

        return this._fetch('/documents/upload', {
            method: 'POST',
            body: formData,
        });
    }

    async deleteDocument(documentId) {
        return this._fetch(`/documents/${documentId}`, {
            method: 'DELETE',
        });
    }

    // ── Knowledge Graph ────────────────────────────────────────
    async getGraphStats() {
        return this._fetch('/graph/stats');
    }

    async getGraphNodes(documentId = null, search = null) {
        const params = new URLSearchParams();
        if (documentId) params.append('document_id', documentId);
        if (search) params.append('search', search);
        const queryStr = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/graph/nodes${queryStr}`);
    }

    async getGraphEdges(documentId = null) {
        const params = new URLSearchParams();
        if (documentId) params.append('document_id', documentId);
        const queryStr = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/graph/edges${queryStr}`);
    }

    async queryGraph(query, maxHops = 1, maxNodes = 50) {
        return this._fetch('/graph/query', {
            method: 'POST',
            body: JSON.stringify({
                query,
                max_hops: maxHops,
                max_nodes: maxNodes
            })
        });
    }

    async getEntitySubgraph(entityName, maxHops = 1, maxNodes = 50) {
        const encoded = encodeURIComponent(entityName);
        return this._fetch(`/graph/subgraph/${encoded}?max_hops=${maxHops}&max_nodes=${maxNodes}`);
    }

    // ── Vector Search ──────────────────────────────────────────
    async vectorSearch(query, topK = 5, documentId = null) {
        return this._fetch('/search', {
            method: 'POST',
            body: JSON.stringify({
                query,
                top_k: topK,
                document_id: documentId
            })
        });
    }

    async getSearchStats() {
        return this._fetch('/search/stats');
    }

    // ── GraphRAG Chat & Intelligence ───────────────────────────
    async queryAssistant({ query, mode = 'hybrid', topKChunks = 3, maxGraphHops = 1, documentId = null }) {
        return this._fetch('/query', {
            method: 'POST',
            body: JSON.stringify({
                query,
                mode,
                top_k_chunks: topKChunks,
                max_graph_hops: maxGraphHops,
                document_id: documentId
            })
        });
    }
}

// Global API instance
window.api = new GraphRagApi();
