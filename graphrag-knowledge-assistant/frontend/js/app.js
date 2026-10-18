/**
 * GraphRAG Knowledge Assistant — Main Application Controller
 * Coordinates UI state, views, chat interactions, graph exploration, and document management.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── Application State ──────────────────────────────────────
    const state = {
        activeTab: 'chat',
        retrievalMode: 'hybrid', // 'hybrid' | 'vector_only' | 'graph_only'
        topKChunks: 3,
        maxGraphHops: 1,
        selectedDocFilter: null,
        documents: [],
        graphData: { nodes: [], edges: [] },
        graphVisualizer: null,
        isLoadingChat: false,
        chatHistory: []
    };
    window.state = state;

    // ── DOM Element Cache ──────────────────────────────────────
    const elements = {
        // Nav
        navTabs: document.querySelectorAll('.nav-tab-btn'),
        viewPanels: document.querySelectorAll('.view-panel'),
        systemStatusDot: document.getElementById('systemStatusDot'),
        systemStatusText: document.getElementById('systemStatusText'),
        statDocsCount: document.getElementById('statDocsCount'),
        statChunksCount: document.getElementById('statChunksCount'),
        statNodesCount: document.getElementById('statNodesCount'),
        statEdgesCount: document.getElementById('statEdgesCount'),

        // Chat View
        chatStream: document.getElementById('chatStream'),
        chatWelcome: document.getElementById('chatWelcome'),
        chatInput: document.getElementById('chatInput'),
        chatSendBtn: document.getElementById('chatSendBtn'),
        modeBtns: document.querySelectorAll('.mode-btn'),
        docFilterSelect: document.getElementById('docFilterSelect'),
        topKSelect: document.getElementById('topKSelect'),
        hopsSelect: document.getElementById('hopsSelect'),
        starterChips: document.querySelectorAll('.starter-chip'),

        // Graph View
        graphCanvas: document.getElementById('graphCanvas'),
        graphSearchInput: document.getElementById('graphSearchInput'),
        typeFilterChips: document.querySelectorAll('.type-filter-chip'),
        btnZoomIn: document.getElementById('btnZoomIn'),
        btnZoomOut: document.getElementById('btnZoomOut'),
        btnFitGraph: document.getElementById('btnFitGraph'),
        btnTogglePhysics: document.getElementById('btnTogglePhysics'),
        graphInspector: document.getElementById('graphInspector'),
        inspectorEmpty: document.getElementById('inspectorEmpty'),
        inspectorContent: document.getElementById('inspectorContent'),
        inspectorEntityName: document.getElementById('inspectorEntityName'),
        inspectorEntityType: document.getElementById('inspectorEntityType'),
        inspectorEntityDesc: document.getElementById('inspectorEntityDesc'),
        inspectorRelationsList: document.getElementById('inspectorRelationsList'),
        inspectorAskBtn: document.getElementById('inspectorAskBtn'),
        inspectorExpandBtn: document.getElementById('inspectorExpandBtn'),

        // Docs View
        dropzone: document.getElementById('dropzone'),
        fileInput: document.getElementById('fileInput'),
        uploadStatusBar: document.getElementById('uploadStatusBar'),
        uploadStatusText: document.getElementById('uploadStatusText'),
        docsTableBody: document.getElementById('docsTableBody'),
        docsEmptyRow: document.getElementById('docsEmptyRow'),

        // Chunks Modal
        chunksModal: document.getElementById('chunksModal'),
        chunksModalTitle: document.getElementById('chunksModalTitle'),
        chunksModalBody: document.getElementById('chunksModalBody'),
        chunksModalClose: document.getElementById('chunksModalClose'),

        // Diagnostics View
        diagApiStatus: document.getElementById('diagApiStatus'),
        diagApiVersion: document.getElementById('diagApiVersion'),
        diagApiEnv: document.getElementById('diagApiEnv'),
        diagVectorCount: document.getElementById('diagVectorCount'),
        diagVectorColl: document.getElementById('diagVectorColl'),
        diagEmbeddingModel: document.getElementById('diagEmbeddingModel'),
        diagGraphNodes: document.getElementById('diagGraphNodes'),
        diagGraphEdges: document.getElementById('diagGraphEdges'),
        diagGraphDensity: document.getElementById('diagGraphDensity'),

        // Toast Container
        toastContainer: document.getElementById('toastContainer')
    };

    // ── Toast Notification Helper ──────────────────────────────
    function showToast(message, type = 'info', duration = 3500) {
        if (!elements.toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const iconClass = type === 'success' ? 'fa-circle-check' :
                          type === 'error' ? 'fa-triangle-exclamation' : 'fa-circle-info';
        
        toast.innerHTML = `<i class="fa-solid ${iconClass}"></i><span>${message}</span>`;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ── Navigation & Tab Switching ─────────────────────────────
    function switchTab(tabId) {
        state.activeTab = tabId;

        // Update tab buttons
        elements.navTabs.forEach(btn => {
            const isActive = btn.dataset.tab === tabId;
            btn.classList.toggle('active', isActive);
        });

        // Update view panels
        elements.viewPanels.forEach(panel => {
            const isActive = panel.id === `view-${tabId}`;
            panel.classList.toggle('active', isActive);
        });

        // Trigger view-specific refreshes
        if (tabId === 'graph') {
            setTimeout(() => {
                if (state.graphVisualizer) {
                    state.graphVisualizer.resize();
                } else {
                    initGraphVisualizer();
                }
            }, 80);
        } else if (tabId === 'docs') {
            loadDocuments();
        } else if (tabId === 'diag') {
            loadDiagnostics();
        }
    }

    elements.navTabs.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // ── System Health & Stats Polling ──────────────────────────
    async function updateSystemStats() {
        try {
            const [health, docsResp, graphStats] = await Promise.all([
                window.api.getHealth(),
                window.api.getDocuments(),
                window.api.getGraphStats()
            ]);

            // Status indicator
            if (elements.systemStatusDot && elements.systemStatusText) {
                elements.systemStatusDot.style.backgroundColor = 'var(--accent-emerald)';
                elements.systemStatusDot.style.boxShadow = '0 0 8px var(--accent-emerald)';
                elements.systemStatusText.textContent = 'System Online';
            }

            // Stats in nav
            const totalDocs = docsResp.total || (docsResp.documents ? docsResp.documents.length : 0);
            let totalChunks = 0;
            if (docsResp.documents) {
                totalChunks = docsResp.documents.reduce((acc, d) => acc + (d.num_chunks ?? d.chunk_count ?? 0), 0);
            }

            if (elements.statDocsCount) elements.statDocsCount.textContent = totalDocs;
            if (elements.statChunksCount) elements.statChunksCount.textContent = totalChunks;
            if (elements.statNodesCount) elements.statNodesCount.textContent = graphStats.num_nodes || 0;
            if (elements.statEdgesCount) elements.statEdgesCount.textContent = graphStats.num_edges || 0;

            state.documents = docsResp.documents || [];
            updateDocFilterDropdowns();

        } catch (error) {
            console.warn('System status fetch failed:', error);
            if (elements.systemStatusDot && elements.systemStatusText) {
                elements.systemStatusDot.style.backgroundColor = 'var(--accent-rose)';
                elements.systemStatusDot.style.boxShadow = '0 0 8px var(--accent-rose)';
                elements.systemStatusText.textContent = 'Backend Offline';
            }
        }
    }

    function updateDocFilterDropdowns() {
        if (!elements.docFilterSelect) return;
        const currentVal = elements.docFilterSelect.value;
        elements.docFilterSelect.innerHTML = '<option value="">All Documents</option>';
        state.documents.forEach(doc => {
            const opt = document.createElement('option');
            opt.value = doc.id;
            opt.textContent = doc.filename;
            elements.docFilterSelect.appendChild(opt);
        });
        elements.docFilterSelect.value = currentVal;
    }

    // ── Chat & Retrieval Logic ─────────────────────────────────

    // Mode Buttons
    elements.modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.retrievalMode = btn.dataset.mode;
        });
    });

    // Param listeners
    if (elements.topKSelect) {
        elements.topKSelect.addEventListener('change', (e) => {
            state.topKChunks = parseInt(e.target.value, 10);
        });
    }
    if (elements.hopsSelect) {
        elements.hopsSelect.addEventListener('change', (e) => {
            state.maxGraphHops = parseInt(e.target.value, 10);
        });
    }
    if (elements.docFilterSelect) {
        elements.docFilterSelect.addEventListener('change', (e) => {
            state.selectedDocFilter = e.target.value || null;
        });
    }

    // Auto-grow textarea
    if (elements.chatInput) {
        elements.chatInput.addEventListener('input', () => {
            elements.chatInput.style.height = 'auto';
            elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 160)}px`;
        });

        elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendChat();
            }
        });
    }

    if (elements.chatSendBtn) {
        elements.chatSendBtn.addEventListener('click', handleSendChat);
    }

    // Starter Prompt Chips
    elements.starterChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            if (query && elements.chatInput) {
                elements.chatInput.value = query;
                handleSendChat();
            }
        });
    });

    async function handleSendChat() {
        if (!elements.chatInput) return;
        const text = elements.chatInput.value.trim();
        if (!text || state.isLoadingChat) return;

        // Hide welcome screen
        if (elements.chatWelcome) {
            elements.chatWelcome.style.display = 'none';
        }

        // Add user message to UI
        appendUserMessage(text);
        elements.chatInput.value = '';
        elements.chatInput.style.height = 'auto';

        // Add loading placeholder message
        const loadingMsgId = appendLoadingMessage();
        state.isLoadingChat = true;
        elements.chatSendBtn.disabled = true;

        const startTime = Date.now();

        try {
            const response = await window.api.queryAssistant({
                query: text,
                mode: state.retrievalMode,
                topKChunks: state.topKChunks,
                maxGraphHops: state.maxGraphHops,
                documentId: state.selectedDocFilter
            });

            const latency = Date.now() - startTime;
            replaceLoadingWithResponse(loadingMsgId, response, latency);

        } catch (error) {
            console.error('Query assistant failed:', error);
            replaceLoadingWithError(loadingMsgId, error.message || 'An error occurred while generating the answer.');
        } finally {
            state.isLoadingChat = false;
            elements.chatSendBtn.disabled = false;
            elements.chatInput.focus();
        }
    }

    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'chat-msg user';
        msgDiv.innerHTML = `
            <div class="msg-body">
                <div class="msg-content">${escapeHtml(text)}</div>
            </div>
            <div class="msg-avatar"><i class="fa-solid fa-user"></i></div>
        `;
        elements.chatStream.appendChild(msgDiv);
        scrollChatToBottom();
    }

    function appendLoadingMessage() {
        const msgId = `loading-${Date.now()}`;
        const msgDiv = document.createElement('div');
        msgDiv.id = msgId;
        msgDiv.className = 'chat-msg assistant';
        msgDiv.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid fa-brain"></i></div>
            <div class="msg-body">
                <div class="msg-content" style="display:flex; align-items:center; gap:10px; color:var(--text-secondary);">
                    <i class="fa-solid fa-spinner fa-spin" style="color:var(--accent-cyan);"></i>
                    <span>Analyzing knowledge graph & retrieving semantic context...</span>
                </div>
            </div>
        `;
        elements.chatStream.appendChild(msgDiv);
        scrollChatToBottom();
        return msgId;
    }

    function replaceLoadingWithResponse(msgId, data, latency) {
        const loadingDiv = document.getElementById(msgId);
        if (!loadingDiv) return;

        // Render Markdown safely
        const rawAnswer = data.answer || 'No answer generated.';
        const parsedMarkdown = (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined')
            ? DOMPurify.sanitize(marked.parse(rawAnswer))
            : escapeHtml(rawAnswer);

        // Sources count
        const docSources = (data.sources || []).filter(s => s.source_type === 'vector' || s.source_type === 'document');
        const graphTriples = data.graph_triples || [];
        const hasSources = (data.sources && data.sources.length > 0) || graphTriples.length > 0;

        const modeClass = data.mode === 'vector_only' ? 'vector' :
                          data.mode === 'graph_only' ? 'graph' : 'hybrid';
        const modeLabel = data.mode === 'vector_only' ? 'Vector Only' :
                          data.mode === 'graph_only' ? 'Graph Only' : 'Hybrid RAG';

        let citationsHtml = '';
        if (hasSources) {
            citationsHtml = `
                <div class="msg-citations-wrap">
                    <button class="citations-accordion-btn" onclick="this.nextElementSibling.classList.toggle('open')">
                        <span><i class="fa-solid fa-book-bookmark" style="margin-right:6px; color:var(--accent-cyan);"></i> Consulted Sources (${(data.sources || []).length} items, ${graphTriples.length} facts)</span>
                        <i class="fa-solid fa-chevron-down"></i>
                    </button>
                    <div class="citations-content">
                        ${graphTriples.length > 0 ? `
                            <div style="margin-top:6px;">
                                <div style="font-size:11px; font-weight:600; color:var(--text-muted); margin-bottom:6px;">ACTIVATED GRAPH TRIPLES</div>
                                <div class="triples-list">
                                    ${graphTriples.map(t => `
                                        <div class="triple-badge">
                                            <i class="fa-solid fa-diagram-project"></i>
                                            <span>${escapeHtml(t.subject)} ➔ <b>${escapeHtml(t.relation)}</b> ➔ ${escapeHtml(t.object)}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}

                        ${docSources.length > 0 ? `
                            <div style="margin-top:8px;">
                                <div style="font-size:11px; font-weight:600; color:var(--text-muted); margin-bottom:6px;">RETRIEVED DOCUMENT PASSAGES</div>
                                <div style="display:flex; flex-direction:column; gap:8px;">
                                    ${docSources.map(s => `
                                        <div class="citation-card">
                                            <div class="citation-card-header">
                                                <span class="citation-filename"><i class="fa-regular fa-file-lines" style="margin-right:4px;"></i> ${escapeHtml(s.filename || 'Document')} (Chunk #${s.chunk_index !== null ? s.chunk_index : '?'})</span>
                                                <span class="citation-score">Relevance: ${Math.round(s.score * 100)}%</span>
                                            </div>
                                            <div class="citation-text">"${escapeHtml((s.content || '').slice(0, 220))}..."</div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}

                        ${data.subgraph && data.subgraph.nodes && data.subgraph.nodes.length > 0 ? `
                            <div style="margin-top:8px; display:flex; justify-content:flex-end;">
                                <button class="btn-secondary" style="font-size:11.5px;" onclick="window.exploreQuerySubgraph('${encodeURIComponent(JSON.stringify(data.subgraph))}')">
                                    <i class="fa-solid fa-circle-nodes" style="color:var(--accent-cyan);"></i> View Active Subgraph in Visualizer
                                </button>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }

        loadingDiv.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid fa-brain"></i></div>
            <div class="msg-body">
                <div class="msg-content">
                    <div class="msg-meta-header">
                        <span class="msg-mode-tag ${modeClass}"><i class="fa-solid fa-bolt"></i> ${modeLabel}</span>
                        <span style="color:var(--text-muted);"><i class="fa-regular fa-clock" style="margin-right:4px;"></i>${latency}ms</span>
                    </div>
                    <div class="markdown-body">${parsedMarkdown}</div>
                    ${citationsHtml}
                </div>
            </div>
        `;

        scrollChatToBottom();
    }

    function replaceLoadingWithError(msgId, errorText) {
        const loadingDiv = document.getElementById(msgId);
        if (!loadingDiv) return;

        loadingDiv.innerHTML = `
            <div class="msg-avatar" style="background:var(--accent-rose);"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="msg-body">
                <div class="msg-content" style="border-color:rgba(244,63,94,0.3); background:rgba(244,63,94,0.06);">
                    <div style="color:var(--accent-rose); font-weight:600; margin-bottom:4px;">Generation Error</div>
                    <div style="color:var(--text-secondary); font-size:13px;">${escapeHtml(errorText)}</div>
                </div>
            </div>
        `;
        scrollChatToBottom();
    }

    function scrollChatToBottom() {
        if (elements.chatStream) {
            elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
        }
    }

    // ── Knowledge Graph Visualizer Integration ─────────────────
    async function initGraphVisualizer() {
        if (!elements.graphCanvas || typeof vis === 'undefined') return;

        if (!state.graphVisualizer) {
            state.graphVisualizer = new GraphVisualizer('graphCanvas', {
                onNodeSelected: (node, neighbors) => {
                    handleNodeClick(node, neighbors);
                }
            });

            // Bind controls
            if (elements.btnZoomIn) elements.btnZoomIn.addEventListener('click', () => state.graphVisualizer.zoomIn());
            if (elements.btnZoomOut) elements.btnZoomOut.addEventListener('click', () => state.graphVisualizer.zoomOut());
            if (elements.btnFitGraph) elements.btnFitGraph.addEventListener('click', () => state.graphVisualizer.fit());
            
            let physicsOn = true;
            if (elements.btnTogglePhysics) {
                elements.btnTogglePhysics.addEventListener('click', () => {
                    physicsOn = !physicsOn;
                    state.graphVisualizer.togglePhysics(physicsOn);
                    elements.btnTogglePhysics.innerHTML = physicsOn 
                        ? '<i class="fa-solid fa-pause"></i>' 
                        : '<i class="fa-solid fa-play"></i>';
                });
            }

            // Search input
            if (elements.graphSearchInput) {
                elements.graphSearchInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        const term = elements.graphSearchInput.value.trim();
                        if (term) {
                            const found = state.graphVisualizer.focusEntity(term);
                            if (!found) {
                                showToast(`Entity "${term}" not found in current graph`, 'info');
                            }
                        }
                    }
                });
            }

            // Type filter chips
            elements.typeFilterChips.forEach(chip => {
                chip.addEventListener('click', () => {
                    elements.typeFilterChips.forEach(c => c.classList.remove('active'));
                    chip.classList.add('active');
                    const type = chip.dataset.type;
                    state.graphVisualizer.filterByType(type);
                });
            });
        }

        // Fetch graph data
        try {
            const [nodes, edges] = await Promise.all([
                window.api.getGraphNodes(),
                window.api.getGraphEdges()
            ]);

            state.graphData = { nodes: nodes || [], edges: edges || [] };
            state.graphVisualizer.setData(nodes, edges);

            // Ensure canvas fits full container after dataset load
            setTimeout(() => {
                if (state.graphVisualizer) {
                    state.graphVisualizer.resize();
                }
            }, 100);

        } catch (error) {
            console.error('Failed to load graph data:', error);
            showToast('Failed to load knowledge graph', 'error');
        }
    }

    // Auto-resize graph on window resize
    window.addEventListener('resize', () => {
        if (state.activeTab === 'graph' && state.graphVisualizer) {
            state.graphVisualizer.resize();
        }
    });

    function handleNodeClick(node, neighbors) {
        if (!elements.inspectorEmpty || !elements.inspectorContent) return;

        if (!node) {
            elements.inspectorEmpty.style.display = 'block';
            elements.inspectorContent.style.display = 'none';
            return;
        }

        elements.inspectorEmpty.style.display = 'none';
        elements.inspectorContent.style.display = 'flex';

        elements.inspectorEntityName.textContent = node.name;
        elements.inspectorEntityType.textContent = node.entity_type || 'CONCEPT';
        elements.inspectorEntityDesc.textContent = node.description || 'No description available for this entity.';

        // List relations
        elements.inspectorRelationsList.innerHTML = '';
        if (neighbors && (neighbors.incoming.length > 0 || neighbors.outgoing.length > 0)) {
            [...neighbors.outgoing, ...neighbors.incoming].forEach(edge => {
                const item = document.createElement('div');
                item.className = 'inspector-rel-item';
                item.innerHTML = `
                    <span>${escapeHtml(edge.source)}</span>
                    <span class="rel-name">-[${escapeHtml(edge.relation)}]-></span>
                    <span>${escapeHtml(edge.target)}</span>
                `;
                elements.inspectorRelationsList.appendChild(item);
            });
        } else {
            elements.inspectorRelationsList.innerHTML = '<div style="color:var(--text-muted); font-size:12px;">No direct connections</div>';
        }

        // Action button: Ask about this entity
        if (elements.inspectorAskBtn) {
            elements.inspectorAskBtn.onclick = () => {
                switchTab('chat');
                if (elements.chatInput) {
                    elements.chatInput.value = `Explain the role of "${node.name}" and how it connects to other concepts in the documents.`;
                    handleSendChat();
                }
            };
        }

        // Action button: Expand 1-hop
        if (elements.inspectorExpandBtn) {
            elements.inspectorExpandBtn.onclick = async () => {
                try {
                    const subgraph = await window.api.getEntitySubgraph(node.name, 2, 50);
                    if (subgraph && subgraph.nodes) {
                        state.graphVisualizer.setData(subgraph.nodes, subgraph.edges);
                        showToast(`Expanded 2-hop neighborhood for "${node.name}"`, 'success');
                    }
                } catch (e) {
                    showToast(`Failed to expand: ${e.message}`, 'error');
                }
            };
        }
    }

    // Global helper called from chat citation to view active subgraph
    window.exploreQuerySubgraph = (encodedSubgraph) => {
        try {
            const subgraph = JSON.parse(decodeURIComponent(encodedSubgraph));
            switchTab('graph');
            setTimeout(() => {
                if (state.graphVisualizer && subgraph.nodes) {
                    state.graphVisualizer.setData(subgraph.nodes, subgraph.edges);
                    showToast(`Displaying ${subgraph.nodes.length} nodes from query retrieval`, 'info');
                }
            }, 100);
        } catch (e) {
            console.error('Failed to parse subgraph:', e);
        }
    };

    // ── Document Management ────────────────────────────────────
    async function loadDocuments() {
        if (!elements.docsTableBody) return;

        try {
            const data = await window.api.getDocuments();
            const docs = data.documents || [];
            state.documents = docs;

            if (docs.length === 0) {
                if (elements.docsEmptyRow) elements.docsEmptyRow.style.display = 'table-row';
                elements.docsTableBody.innerHTML = '';
                elements.docsTableBody.appendChild(elements.docsEmptyRow);
                return;
            }

            if (elements.docsEmptyRow) elements.docsEmptyRow.style.display = 'none';
            elements.docsTableBody.innerHTML = '';

            docs.forEach(doc => {
                const tr = document.createElement('tr');
                const fileSizeKb = Math.round((doc.file_size || 0) / 1024);
                const dateStr = doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'Recent';

                tr.innerHTML = `
                    <td>
                        <div class="doc-name-cell">
                            <i class="fa-regular fa-file-lines doc-icon"></i>
                            <div>
                                <div>${escapeHtml(doc.filename)}</div>
                                <div style="font-size:11px; color:var(--text-muted);">${doc.file_type || 'TXT'}</div>
                            </div>
                        </div>
                    </td>
                    <td><span class="stat-pill" style="display:inline-flex;"><b>${doc.num_chunks ?? doc.chunk_count ?? 0}</b> chunks</span></td>
                    <td style="color:var(--text-secondary);">${fileSizeKb} KB</td>
                    <td style="color:var(--text-secondary);">${dateStr}</td>
                    <td>
                        <div class="doc-actions">
                            <button class="btn-secondary" onclick="window.viewDocChunks('${doc.id}')">
                                <i class="fa-solid fa-list-check"></i> Chunks
                            </button>
                            <button class="btn-danger" onclick="window.confirmDeleteDoc('${doc.id}', '${escapeHtml(doc.filename)}')">
                                <i class="fa-regular fa-trash-can"></i>
                            </button>
                        </div>
                    </td>
                `;
                elements.docsTableBody.appendChild(tr);
            });

        } catch (error) {
            console.error('Failed to load documents:', error);
            showToast('Failed to load documents list', 'error');
        }
    }

    // Dropzone Upload
    if (elements.dropzone && elements.fileInput) {
        elements.dropzone.addEventListener('click', () => elements.fileInput.click());

        elements.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.dropzone.classList.add('drag-over');
        });

        elements.dropzone.addEventListener('dragleave', () => {
            elements.dropzone.classList.remove('drag-over');
        });

        elements.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.dropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        elements.fileInput.addEventListener('change', () => {
            if (elements.fileInput.files.length > 0) {
                handleFileUpload(elements.fileInput.files[0]);
            }
        });
    }

    async function handleFileUpload(file) {
        if (!file) return;

        // Show upload indicator
        if (elements.uploadStatusBar && elements.uploadStatusText) {
            elements.uploadStatusBar.classList.add('active');
            elements.uploadStatusText.innerHTML = `
                <i class="fa-solid fa-spinner fa-spin" style="color:var(--accent-cyan); margin-right:8px;"></i>
                Uploading <b>${escapeHtml(file.name)}</b> & executing GraphRAG Ingestion Pipeline (Chunking, Vector Embeddings, NER & Knowledge Graph Extraction)...
            `;
        }

        try {
            const resp = await window.api.uploadDocument(file);
            showToast(`Document "${file.name}" ingested successfully! (${resp.chunks_created || 0} chunks, ${resp.entities_extracted || 0} entities)`, 'success', 5000);
            
            // Refresh documents and stats
            await Promise.all([loadDocuments(), updateSystemStats()]);

            // If graph visualizer is active, refresh it
            if (state.graphVisualizer) {
                const [nodes, edges] = await Promise.all([
                    window.api.getGraphNodes(),
                    window.api.getGraphEdges()
                ]);
                state.graphVisualizer.setData(nodes, edges);
            }

        } catch (error) {
            console.error('Upload failed:', error);
            showToast(`Ingestion failed: ${error.message}`, 'error', 6000);
        } finally {
            if (elements.uploadStatusBar) {
                elements.uploadStatusBar.classList.remove('active');
            }
            if (elements.fileInput) elements.fileInput.value = '';
        }
    }

    // View Document Chunks Modal
    window.viewDocChunks = async (documentId) => {
        if (!elements.chunksModal || !elements.chunksModalBody) return;

        elements.chunksModal.classList.add('open');
        elements.chunksModalTitle.textContent = 'Loading Document Chunks...';
        elements.chunksModalBody.innerHTML = '<div style="text-align:center; padding:40px;"><i class="fa-solid fa-spinner fa-spin fa-2x" style="color:var(--accent-cyan);"></i></div>';

        try {
            const data = await window.api.getDocumentChunks(documentId);
            elements.chunksModalTitle.innerHTML = `<i class="fa-regular fa-file-lines" style="color:var(--accent-indigo); margin-right:8px;"></i> ${escapeHtml(data.filename)} (${data.total_chunks} Chunks)`;

            elements.chunksModalBody.innerHTML = '';
            (data.chunks || []).forEach(chunk => {
                const card = document.createElement('div');
                card.className = 'citation-card';
                card.style.padding = '14px 16px';
                card.innerHTML = `
                    <div class="citation-card-header" style="margin-bottom:8px;">
                        <span style="font-weight:600; color:var(--accent-cyan); font-family:var(--font-mono);">
                            Chunk #${chunk.chunk_index}
                        </span>
                        <span style="color:var(--text-muted); font-size:11px;">
                            ${(chunk.content || '').length} characters • ${chunk.token_count || 0} tokens
                        </span>
                    </div>
                    <div style="font-size:12.5px; line-height:1.6; color:var(--text-primary); white-space:pre-wrap; background:rgba(0,0,0,0.25); padding:10px; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);">${escapeHtml(chunk.content)}</div>
                `;
                elements.chunksModalBody.appendChild(card);
            });

        } catch (error) {
            elements.chunksModalBody.innerHTML = `<div style="color:var(--accent-rose); padding:20px;">Failed to load chunks: ${escapeHtml(error.message)}</div>`;
        }
    };

    if (elements.chunksModalClose) {
        elements.chunksModalClose.addEventListener('click', () => {
            elements.chunksModal.classList.remove('open');
        });
    }

    // Delete Document
    window.confirmDeleteDoc = async (documentId, filename) => {
        if (!confirm(`Are you sure you want to delete "${filename}"?\nThis will remove all text chunks, vector embeddings, and graph entities extracted from this file.`)) {
            return;
        }

        try {
            await window.api.deleteDocument(documentId);
            showToast(`Document "${filename}" and its knowledge representations deleted.`, 'info');
            await Promise.all([loadDocuments(), updateSystemStats()]);

            // Refresh graph if initialized
            if (state.graphVisualizer) {
                const [nodes, edges] = await Promise.all([
                    window.api.getGraphNodes(),
                    window.api.getGraphEdges()
                ]);
                state.graphVisualizer.setData(nodes, edges);
            }
        } catch (error) {
            showToast(`Delete failed: ${error.message}`, 'error');
        }
    };

    // ── Diagnostics View ───────────────────────────────────────
    async function loadDiagnostics() {
        try {
            const [health, searchStats, graphStats] = await Promise.all([
                window.api.getHealth(),
                window.api.getSearchStats(),
                window.api.getGraphStats()
            ]);

            if (elements.diagApiStatus) elements.diagApiStatus.textContent = (health.status || 'unknown').toUpperCase();
            if (elements.diagApiVersion) elements.diagApiVersion.textContent = health.version || '0.1.0';
            if (elements.diagApiEnv) elements.diagApiEnv.textContent = health.environment || 'development';

            if (elements.diagVectorCount) elements.diagVectorCount.textContent = searchStats.total_vectors ?? searchStats.total_chunks ?? 0;
            if (elements.diagVectorColl) elements.diagVectorColl.textContent = searchStats.collection_name || 'document_chunks';
            if (elements.diagEmbeddingModel) elements.diagEmbeddingModel.textContent = searchStats.embedding_model || 'all-MiniLM-L6-v2';

            if (elements.diagGraphNodes) elements.diagGraphNodes.textContent = graphStats.num_nodes || 0;
            if (elements.diagGraphEdges) elements.diagGraphEdges.textContent = graphStats.num_edges || 0;
            if (elements.diagGraphDensity) elements.diagGraphDensity.textContent = (graphStats.density !== undefined ? graphStats.density.toFixed(4) : '0.0000');

        } catch (error) {
            console.error('Failed to load diagnostics:', error);
        }
    }

    // ── Utility: Escape HTML ───────────────────────────────────
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ── Initial Boot ───────────────────────────────────────────
    updateSystemStats();
    // Poll stats every 30 seconds
    setInterval(updateSystemStats, 30000);
});
