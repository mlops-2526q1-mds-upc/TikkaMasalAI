"""
Simple metrics dashboard for Prometheus metrics.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from prometheus_client import REGISTRY
import json

router = APIRouter()

@router.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard():
    """
    Simple visual dashboard for Prometheus metrics.
    No external dependencies - just HTML + JavaScript.
    """
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tikka MasalAI - Metrics Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            h1 {
                color: white;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .metric-card {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            
            .metric-title {
                font-size: 14px;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }
            
            .metric-value {
                font-size: 36px;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .metric-subtitle {
                font-size: 12px;
                color: #999;
            }
            
            .table-card {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .table-card h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
            }
            
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            
            th {
                background: #f8f9fa;
                font-weight: 600;
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            tr:hover {
                background: #f8f9fa;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
            }
            
            .status-success {
                background: #d4edda;
                color: #155724;
            }
            
            .status-warning {
                background: #fff3cd;
                color: #856404;
            }
            
            .refresh-info {
                text-align: center;
                color: white;
                margin-top: 20px;
                font-size: 14px;
            }
            
            .loading {
                text-align: center;
                color: white;
                font-size: 18px;
                padding: 50px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍛 Tikka MasalAI Metrics Dashboard</h1>
            
            <div id="dashboard-content" class="loading">
                Loading metrics...
            </div>
            
            <div class="refresh-info">
                Auto-refreshing every 10 seconds
            </div>
        </div>
        
        <script>
            async function fetchMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const text = await response.text();
                    return parsePrometheusMetrics(text);
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                    return null;
                }
            }
            
            function parsePrometheusMetrics(text) {
                const lines = text.split('\\n');
                const metrics = {};
                
                for (const line of lines) {
                    if (line.startsWith('#') || line.trim() === '') continue;
                    
                    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\\{[^}]*\\})?\\s+(.+)$/);
                    if (match) {
                        const metricName = match[1];
                        const labels = match[2] || '';
                        const value = parseFloat(match[3]);
                        
                        if (!metrics[metricName]) {
                            metrics[metricName] = [];
                        }
                        metrics[metricName].push({ labels, value });
                    }
                }
                
                return metrics;
            }
            
            function calculateTotalRequests(metrics) {
                if (!metrics.http_requests_total) return 0;
                return metrics.http_requests_total.reduce((sum, m) => sum + m.value, 0);
            }
            
            function calculateAverageLatency(metrics) {
                if (!metrics.http_request_duration_seconds_sum || !metrics.http_request_duration_seconds_count) {
                    return 0;
                }
                
                const totalTime = metrics.http_request_duration_seconds_sum.reduce((sum, m) => sum + m.value, 0);
                const totalCount = metrics.http_request_duration_seconds_count.reduce((sum, m) => sum + m.value, 0);
                
                return totalCount > 0 ? (totalTime / totalCount * 1000).toFixed(2) : 0;
            }
            
            function getRequestsByEndpoint(metrics) {
                if (!metrics.http_requests_total) return [];
                
                const endpointMap = {};
                
                for (const metric of metrics.http_requests_total) {
                    const handlerMatch = metric.labels.match(/handler="([^"]+)"/);
                    const methodMatch = metric.labels.match(/method="([^"]+)"/);
                    const statusMatch = metric.labels.match(/status="([^"]+)"/);
                    
                    if (handlerMatch && methodMatch) {
                        const key = `${methodMatch[1]} ${handlerMatch[1]}`;
                        const status = statusMatch ? statusMatch[1] : 'unknown';
                        
                        if (!endpointMap[key]) {
                            endpointMap[key] = { endpoint: key, requests: 0, status: status };
                        }
                        endpointMap[key].requests += metric.value;
                    }
                }
                
                return Object.values(endpointMap).sort((a, b) => b.requests - a.requests);
            }
            
            function getPythonVersion(metrics) {
                if (!metrics.python_info) return 'Unknown';
                
                for (const metric of metrics.python_info) {
                    const majorMatch = metric.labels.match(/major="([^"]+)"/);
                    const minorMatch = metric.labels.match(/minor="([^"]+)"/);
                    const patchMatch = metric.labels.match(/patchlevel="([^"]+)"/);
                    
                    if (majorMatch && minorMatch && patchMatch) {
                        return `${majorMatch[1]}.${minorMatch[1]}.${patchMatch[1]}`;
                    }
                }
                return 'Unknown';
            }
            
            function renderDashboard(metrics) {
                if (!metrics) {
                    document.getElementById('dashboard-content').innerHTML = 
                        '<div class="loading">Failed to load metrics. Please check if /metrics endpoint is available.</div>';
                    return;
                }
                
                const totalRequests = calculateTotalRequests(metrics);
                const avgLatency = calculateAverageLatency(metrics);
                const pythonVersion = getPythonVersion(metrics);
                const endpoints = getRequestsByEndpoint(metrics);
                
                const html = `
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-title">Total Requests</div>
                            <div class="metric-value">${totalRequests}</div>
                            <div class="metric-subtitle">Since startup</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">Avg Response Time</div>
                            <div class="metric-value">${avgLatency} ms</div>
                            <div class="metric-subtitle">Per request</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">Python Version</div>
                            <div class="metric-value" style="font-size: 28px;">${pythonVersion}</div>
                            <div class="metric-subtitle">CPython</div>
                        </div>
                    </div>
                    
                    <div class="table-card">
                        <h2>📊 Requests by Endpoint</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Endpoint</th>
                                    <th>Requests</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${endpoints.map(ep => `
                                    <tr>
                                        <td><strong>${ep.endpoint}</strong></td>
                                        <td>${ep.requests}</td>
                                        <td><span class="status-badge ${ep.status.startsWith('2') ? 'status-success' : 'status-warning'}">${ep.status}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
                document.getElementById('dashboard-content').innerHTML = html;
            }
            
            async function updateDashboard() {
                const metrics = await fetchMetrics();
                renderDashboard(metrics);
            }
            
            // Initial load
            updateDashboard();
            
            // Auto-refresh every 10 seconds
            setInterval(updateDashboard, 10000);
        </script>
    </body>
    </html>
    """
    
    return html_content