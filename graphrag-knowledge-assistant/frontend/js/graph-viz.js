/**
 * GraphRAG Knowledge Assistant — Knowledge Graph Visualizer
 * Powered by Vis.js Network with custom interactive physics, dark-mode theme,
 * entity type color mapping, and neighbor inspection.
 */

class GraphVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = options;
        this.network = null;
        this.nodesDataSet = new vis.DataSet([]);
        this.edgesDataSet = new vis.DataSet([]);
        this.rawNodes = [];
        this.rawEdges = [];
        this.activeFilter = null;
        this.onNodeSelected = options.onNodeSelected || null;
        this.onEdgeSelected = options.onEdgeSelected || null;

        // Color palette tailored for sleek dark mode
        this.typeColors = {
            'CONCEPT': { bg: '#06b6d4', border: '#22d3ee', highlight: '#67e8f9' },
            'TECHNOLOGY': { bg: '#6366f1', border: '#818cf8', highlight: '#a5b4fc' },
            'PERSON': { bg: '#10b981', border: '#34d399', highlight: '#6ee7b7' },
            'ORGANIZATION': { bg: '#f59e0b', border: '#fbbf24', highlight: '#fde68a' },
            'FIELD': { bg: '#a855f7', border: '#c084fc', highlight: '#e9d5ff' },
            'METRIC': { bg: '#ec4899', border: '#f472b6', highlight: '#fbcfe8' },
            'DEFAULT': { bg: '#8b5cf6', border: '#a78bfa', highlight: '#ddd6fe' }
        };

        this._initNetwork();
    }

    _getColorForType(entityType) {
        const type = (entityType || '').toUpperCase();
        return this.typeColors[type] || this.typeColors['DEFAULT'];
    }

    _initNetwork() {
        if (!this.container) return;

        const data = {
            nodes: this.nodesDataSet,
            edges: this.edgesDataSet
        };

        const config = {
            autoResize: true,
            width: '100%',
            height: '100%',
            nodes: {
                shape: 'dot',
                size: 18,
                font: {
                    face: 'Inter, sans-serif',
                    size: 12,
                    color: '#f1f5f9',
                    strokeWidth: 2,
                    strokeColor: '#0b0f19'
                },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.5)',
                    size: 10,
                    x: 0,
                    y: 4
                }
            },
            edges: {
                width: 1.8,
                color: {
                    color: 'rgba(148, 163, 184, 0.45)',
                    highlight: '#38bdf8',
                    hover: '#67e8f9'
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.7
                    }
                },
                font: {
                    face: 'Inter, sans-serif',
                    size: 10,
                    color: '#94a3b8',
                    strokeWidth: 2,
                    strokeColor: '#0f172a',
                    align: 'middle'
                },
                smooth: {
                    enabled: true,
                    type: 'dynamic',
                    roundness: 0.5
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 150,
                navigationButtons: false,
                keyboard: false,
                zoomView: true,
                dragView: true
            },
            physics: {
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.012,
                    springLength: 100,
                    springConstant: 0.07,
                    damping: 0.45,
                    avoidOverlap: 0.85
                },
                stabilization: {
                    enabled: true,
                    iterations: 80,
                    updateInterval: 15,
                    fit: true
                }
            }
        };

        this.network = new vis.Network(this.container, data, config);

        // Auto-fit after initial stabilization
        this.network.on('stabilizationIterationsDone', () => {
            if (this.network) {
                this.network.setOptions({ physics: { enabled: true } });
                this.fit();
            }
        });

        // Bind events
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const nodeData = this.rawNodes.find(n => n.name === nodeId || n.id === nodeId);
                if (this.onNodeSelected && nodeData) {
                    this.onNodeSelected(nodeData, this._getNodeNeighbors(nodeId));
                }
            } else if (params.edges.length > 0) {
                const edgeId = params.edges[0];
                const edgeData = this.edgesDataSet.get(edgeId);
                if (this.onEdgeSelected && edgeData) {
                    this.onEdgeSelected(edgeData);
                }
            } else {
                if (this.onNodeSelected) {
                    this.onNodeSelected(null, null);
                }
            }
        });

        // Hover cursor styling
        this.network.on('hoverNode', () => {
            this.container.style.cursor = 'pointer';
        });
        this.network.on('blurNode', () => {
            this.container.style.cursor = 'default';
        });
    }

    _getNodeNeighbors(nodeId) {
        const connectedEdges = this.rawEdges.filter(
            e => e.source === nodeId || e.target === nodeId
        );
        const incoming = connectedEdges.filter(e => e.target === nodeId);
        const outgoing = connectedEdges.filter(e => e.source === nodeId);
        return { incoming, outgoing, total: connectedEdges.length };
    }

    /**
     * Render graph data
     */
    setData(nodes, edges) {
        this.rawNodes = nodes || [];
        this.rawEdges = edges || [];

        const formattedNodes = this.rawNodes.map(node => {
            const colors = this._getColorForType(node.entity_type);
            const degree = this.rawEdges.filter(e => e.source === node.name || e.target === node.name).length;
            const size = Math.min(36, Math.max(16, 14 + degree * 2.5));

            const tooltipEl = document.createElement('div');
            tooltipEl.style.padding = '8px 12px';
            tooltipEl.style.fontFamily = 'Inter, sans-serif';
            tooltipEl.style.fontSize = '12px';
            tooltipEl.style.lineHeight = '1.5';
            tooltipEl.innerHTML = `
                <div style="font-weight:700; color:${colors.border}; font-size:13px; margin-bottom:3px;">${node.name}</div>
                <div style="color:#94a3b8; font-size:11px;">Type: <span style="color:#f8fafc; font-weight:600;">${node.entity_type || 'Concept'}</span></div>
                <div style="color:#94a3b8; font-size:11px;">Connections: <span style="color:#38bdf8; font-weight:600;">${degree}</span></div>
            `;

            return {
                id: node.name,
                label: node.name,
                title: tooltipEl,
                color: {
                    background: colors.bg,
                    border: colors.border,
                    highlight: {
                        background: colors.highlight,
                        border: '#ffffff'
                    },
                    hover: {
                        background: colors.highlight,
                        border: colors.border
                    }
                },
                size: size,
                entityType: node.entity_type || 'CONCEPT',
                degree: degree,
                raw: node
            };
        });

        const formattedEdges = this.rawEdges.map((edge, index) => {
            return {
                id: `edge-${index}-${edge.source}-${edge.target}`,
                from: edge.source,
                to: edge.target,
                label: edge.relation || '',
                title: `${edge.source} ➔ ${edge.relation} ➔ ${edge.target}`,
                raw: edge
            };
        });

        this.allFormattedNodes = formattedNodes;
        this.allFormattedEdges = formattedEdges;

        this.nodesDataSet.clear();
        this.edgesDataSet.clear();
        this.nodesDataSet.add(formattedNodes);
        this.edgesDataSet.add(formattedEdges);

        this.resize();
    }

    /**
     * Highlight a specific entity and focus camera on it
     */
    focusEntity(entityName) {
        if (!this.network) return false;

        const node = this.nodesDataSet.get(entityName);
        if (!node) return false;

        this.network.selectNodes([entityName]);
        this.network.focus(entityName, {
            scale: 1.3,
            animation: {
                duration: 800,
                easingFunction: 'easeInOutQuad'
            }
        });

        const nodeData = this.rawNodes.find(n => n.name === entityName);
        if (this.onNodeSelected && nodeData) {
            this.onNodeSelected(nodeData, this._getNodeNeighbors(entityName));
        }

        return true;
    }

    /**
     * Filter nodes by entity type safely without breaking Vis.js physics canvas
     */
    filterByType(entityType) {
        this.activeFilter = entityType;
        if (!this.allFormattedNodes || this.allFormattedNodes.length === 0) return;

        let visibleNodes = this.allFormattedNodes;
        if (entityType && entityType !== 'ALL') {
            visibleNodes = this.allFormattedNodes.filter(node => 
                (node.entityType || '').toUpperCase() === entityType.toUpperCase()
            );
        }

        const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
        const visibleEdges = (this.allFormattedEdges || []).filter(edge => 
            visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to)
        );

        this.nodesDataSet.clear();
        this.edgesDataSet.clear();
        this.nodesDataSet.add(visibleNodes);
        this.edgesDataSet.add(visibleEdges);

        this.resize();
    }

    /**
     * Resize canvas and fit nodes
     */
    resize() {
        if (this.network && this.container) {
            const w = this.container.clientWidth;
            const h = this.container.clientHeight;
            if (w > 0 && h > 0) {
                this.network.setSize(`${w}px`, `${h}px`);
            } else {
                this.network.setSize('100%', '100%');
            }
            this.network.redraw();
            setTimeout(() => {
                if (this.network) {
                    this.network.fit({
                        animation: {
                            duration: 500,
                            easingFunction: 'easeOutQuad'
                        }
                    });
                }
            }, 80);
        }
    }

    /**
     * Fit all visible nodes into view
     */
    fit() {
        if (this.network) {
            this.network.fit({
                animation: {
                    duration: 600,
                    easingFunction: 'easeOutQuad'
                }
            });
        }
    }

    /**
     * Zoom controls
     */
    zoomIn() {
        if (this.network) {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale * 1.3, animation: { duration: 300 } });
        }
    }

    zoomOut() {
        if (this.network) {
            const scale = this.network.getScale();
            this.network.moveTo({ scale: scale / 1.3, animation: { duration: 300 } });
        }
    }

    togglePhysics(enabled) {
        if (this.network) {
            this.network.setOptions({ physics: { enabled } });
        }
    }

    destroy() {
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }
}

window.GraphVisualizer = GraphVisualizer;
