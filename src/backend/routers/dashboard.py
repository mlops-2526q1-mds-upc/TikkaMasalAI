"""
Enhanced metrics dashboard for FastAPI with charts.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/dashboard", response_class=HTMLResponse)
async def metrics_dashboard():
    """
    Enhanced visual dashboard for Prometheus metrics with charts.
    Includes Chart.js for visualizations.
    """
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tikka MasalAI - Metrics Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
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
                max-width: 1400px;
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
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
                font-weight: 600;
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
            
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .chart-card {
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .chart-card h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.3em;
            }
            
            .chart-container {
                position: relative;
                height: 300px;
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
                color: #333;  /* Fixed: Added black color */
            }
            
            th {
                background: #f8f9fa;
                font-weight: 600;
                color: #666;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            td {
                color: #1a1a1a;  /* Fixed: Darker color for better visibility */
                font-size: 14px;
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
            
            .status-error {
                background: #f8d7da;
                color: #721c24;
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
            
            .timestamp {
                text-align: center;
                color: white;
                margin-top: 10px;
                font-size: 12px;
                opacity: 0.8;
            }
            
            @media (max-width: 768px) {
                .charts-grid {
                    grid-template-columns: 1fr;
                }
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
            <div class="timestamp" id="last-update"></div>
        </div>
        
        <script>
            let requestsChart = null;
            let latencyChart = null;
            let statusChart = null;
            
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
                            endpointMap[key] = { 
                                endpoint: key, 
                                method: methodMatch[1],
                                path: handlerMatch[1],
                                requests: 0, 
                                status: status 
                            };
                        }
                        endpointMap[key].requests += metric.value;
                    }
                }
                
                return Object.values(endpointMap).sort((a, b) => b.requests - a.requests);
            }
            
            function getStatusCodeDistribution(metrics) {
                const distribution = { '2xx': 0, '3xx': 0, '4xx': 0, '5xx': 0 };
                
                if (!metrics.http_requests_total) return distribution;
                
                for (const metric of metrics.http_requests_total) {
                    const statusMatch = metric.labels.match(/status="([^"]+)"/);
                    if (statusMatch) {
                        const status = statusMatch[1];
                        if (distribution[status] !== undefined) {
                            distribution[status] += metric.value;
                        }
                    }
                }
                
                return distribution;
            }
            
            function getLatencyByEndpoint(metrics) {
                if (!metrics.http_request_duration_seconds_sum || !metrics.http_request_duration_seconds_count) {
                    return [];
                }
                
                const latencyMap = {};
                
                // Sum durations by handler
                for (const metric of metrics.http_request_duration_seconds_sum) {
                    const handlerMatch = metric.labels.match(/handler="([^"]+)"/);
                    if (handlerMatch) {
                        const handler = handlerMatch[1];
                        if (!latencyMap[handler]) {
                            latencyMap[handler] = { sum: 0, count: 0 };
                        }
                        latencyMap[handler].sum += metric.value;
                    }
                }
                
                // Count requests by handler
                for (const metric of metrics.http_request_duration_seconds_count) {
                    const handlerMatch = metric.labels.match(/handler="([^"]+)"/);
                    if (handlerMatch) {
                        const handler = handlerMatch[1];
                        if (latencyMap[handler]) {
                            latencyMap[handler].count += metric.value;
                        }
                    }
                }
                
                // Calculate averages
                return Object.entries(latencyMap).map(([handler, data]) => ({
                    endpoint: handler,
                    avgLatency: data.count > 0 ? (data.sum / data.count * 1000).toFixed(2) : 0
                })).sort((a, b) => b.avgLatency - a.avgLatency);
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
            
            function createRequestsChart(endpoints) {
                const ctx = document.getElementById('requestsChart');
                if (!ctx) return;
                
                // Destroy existing chart
                if (requestsChart) {
                    requestsChart.destroy();
                }
                
                const labels = endpoints.map(ep => ep.path);
                const data = endpoints.map(ep => ep.requests);
                
                requestsChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Requests',
                            data: data,
                            backgroundColor: [
                                'rgba(102, 126, 234, 0.8)',
                                'rgba(118, 75, 162, 0.8)',
                                'rgba(237, 100, 166, 0.8)',
                                'rgba(255, 154, 158, 0.8)',
                                'rgba(250, 208, 196, 0.8)'
                            ],
                            borderColor: [
                                'rgba(102, 126, 234, 1)',
                                'rgba(118, 75, 162, 1)',
                                'rgba(237, 100, 166, 1)',
                                'rgba(255, 154, 158, 1)',
                                'rgba(250, 208, 196, 1)'
                            ],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            },
                            title: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    precision: 0
                                }
                            }
                        }
                    }
                });
            }
            
            function createLatencyChart(latencyData) {
                const ctx = document.getElementById('latencyChart');
                if (!ctx) return;
                
                // Destroy existing chart
                if (latencyChart) {
                    latencyChart.destroy();
                }
                
                const labels = latencyData.map(item => item.endpoint);
                const data = latencyData.map(item => parseFloat(item.avgLatency));
                
                latencyChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Avg Latency (ms)',
                            data: data,
                            backgroundColor: 'rgba(102, 126, 234, 0.2)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 5,
                            pointBackgroundColor: 'rgba(102, 126, 234, 1)'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Milliseconds'
                                }
                            }
                        }
                    }
                });
            }
            
            function createStatusChart(statusDistribution) {
                const ctx = document.getElementById('statusChart');
                if (!ctx) return;
                
                // Destroy existing chart
                if (statusChart) {
                    statusChart.destroy();
                }
                
                const labels = Object.keys(statusDistribution);
                const data = Object.values(statusDistribution);
                
                statusChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: [
                                'rgba(40, 167, 69, 0.8)',   // 2xx - green
                                'rgba(23, 162, 184, 0.8)',  // 3xx - cyan
                                'rgba(255, 193, 7, 0.8)',   // 4xx - yellow
                                'rgba(220, 53, 69, 0.8)'    // 5xx - red
                            ],
                            borderColor: [
                                'rgba(40, 167, 69, 1)',
                                'rgba(23, 162, 184, 1)',
                                'rgba(255, 193, 7, 1)',
                                'rgba(220, 53, 69, 1)'
                            ],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
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
                const latencyData = getLatencyByEndpoint(metrics);
                const statusDistribution = getStatusCodeDistribution(metrics);
                
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
                            <div class="metric-title">Active Endpoints</div>
                            <div class="metric-value">${endpoints.length}</div>
                            <div class="metric-subtitle">Serving requests</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">Python Version</div>
                            <div class="metric-value" style="font-size: 28px;">${pythonVersion}</div>
                            <div class="metric-subtitle">CPython</div>
                        </div>
                    </div>
                    
                    <div class="charts-grid">
                        <div class="chart-card">
                            <h2>📊 Requests by Endpoint</h2>
                            <div class="chart-container">
                                <canvas id="requestsChart"></canvas>
                            </div>
                        </div>
                        
                        <div class="chart-card">
                            <h2>⚡ Average Latency</h2>
                            <div class="chart-container">
                                <canvas id="latencyChart"></canvas>
                            </div>
                        </div>
                        
                        <div class="chart-card">
                            <h2>✅ Status Code Distribution</h2>
                            <div class="chart-container">
                                <canvas id="statusChart"></canvas>
                            </div>
                        </div>
                    </div>
                    
                    <div class="table-card">
                        <h2>📋 Detailed Endpoint Statistics</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Method</th>
                                    <th>Endpoint</th>
                                    <th>Requests</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${endpoints.map(ep => `
                                    <tr>
                                        <td><strong>${ep.method}</strong></td>
                                        <td>${ep.path}</td>
                                        <td>${ep.requests}</td>
                                        <td><span class="status-badge ${
                                            ep.status.startsWith('2') ? 'status-success' : 
                                            ep.status.startsWith('4') ? 'status-warning' : 
                                            'status-error'
                                        }">${ep.status}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                
                document.getElementById('dashboard-content').innerHTML = html;
                
                // Create charts after DOM is updated
                setTimeout(() => {
                    createRequestsChart(endpoints);
                    createLatencyChart(latencyData);
                    createStatusChart(statusDistribution);
                }, 100);
                
                // Update timestamp
                const now = new Date();
                document.getElementById('last-update').textContent = 
                    `Last updated: ${now.toLocaleTimeString()}`;
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
