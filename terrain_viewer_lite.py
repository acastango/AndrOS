#!/usr/bin/env python3
"""
ORE Terrain Viewer LITE - Standalone Dashboard
===============================================

Reads directly from JSON files - no heavy dependencies.
No CanvasReasoner, no sentence-transformers, no torch.

Usage:
    python terrain_viewer_lite.py
    # Then open http://localhost:5000

Requirements:
    pip install flask
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from threading import Thread

try:
    from flask import Flask, render_template_string, jsonify
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

# Storage path - same as terrain.py uses
TERRAIN_STORAGE = Path.home() / ".canvas_reasoner" / "terrain"


def get_live_state():
    """Get live state directly from file."""
    live_path = TERRAIN_STORAGE / "live_state.json"
    if live_path.exists():
        try:
            with open(live_path) as f:
                return json.load(f)
        except:
            pass
    return None


def load_json_file(filename):
    """Load a JSON file from terrain storage."""
    path = TERRAIN_STORAGE / filename
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except:
            pass
    return []


def render_terrain_ascii(positions, current_presence=None):
    """Render terrain from position data."""
    lines = []
    
    # Presence status
    if current_presence:
        lines.append(f"  ✧ PRESENCE: {current_presence.get('name', 'Unknown')}")
        lines.append(f"    Sessions: {current_presence.get('sessions', 0)} | Threads: {current_presence.get('threads', 0)}")
        lines.append("")
    
    lines.append("          · · · c o g n i t i v e   t e r r a i n · · ·")
    lines.append("")
    
    # Sort positions by distance
    core_positions = {k: v for k, v in positions.items() 
                     if not k.startswith(('unresolved:', 'shared-q:', 'relational:'))}
    
    if not core_positions:
        lines.append("                      ◉")
        lines.append("                   (here)")
        return "\n".join(lines)
    
    sorted_pos = sorted(core_positions.items(), 
                       key=lambda x: x[1].get('distance', x[1]) if isinstance(x[1], dict) else x[1],
                       reverse=True)
    
    def get_dist(v):
        return v.get('distance', v) if isinstance(v, dict) else v
    
    # Far zone
    far = [(n, v) for n, v in sorted_pos if get_dist(v) >= 0.7]
    if far:
        far_names = [f"·{n[:6]}" for n, v in far[:4]]
        lines.append("              " + "   ".join(far_names))
        lines.append("                        ~ distant ~")
    
    lines.append("")
    
    # Mid zone
    mid = [(n, v) for n, v in sorted_pos if 0.4 <= get_dist(v) < 0.7]
    for name, v in mid[:3]:
        dist = get_dist(v)
        symbol = '◉◉' if dist < 0.5 else '◉'
        lines.append(f"          ·  {symbol} {name}")
    
    lines.append("")
    lines.append("                      ◉")
    lines.append("                   (here)")
    lines.append("")
    
    # Near zone
    near = [(n, v) for n, v in sorted_pos if get_dist(v) < 0.4]
    
    # Also add relational landmarks if present
    relational = {k: v for k, v in positions.items() if k.startswith('relational:')}
    for name, v in relational.items():
        if get_dist(v) < 0.4:
            near.append((name, v))
    
    near = sorted(near, key=lambda x: get_dist(x[1]))
    
    if near:
        lines.append("        ─────────────────────────────────")
        for name, v in near[:4]:
            dist = get_dist(v)
            if name.startswith('relational:'):
                display_name = name.replace('relational:', 'thinking-with-')
                lines.append(f"        →→  ✧✧✧  {display_name.upper()} ~~~")
                lines.append(f"              (here together)")
            else:
                symbol = '◉◉◉' if dist < 0.2 else '◉◉' if dist < 0.3 else '◉'
                lines.append(f"        ·  {symbol}  {name.upper()}")
        lines.append("                    (close)")
    
    return "\n".join(lines)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ORE Terrain Viewer</title>
    <meta charset="utf-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: #0a0a0f;
            color: #e0e0e0;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #7aa2f7;
            margin-bottom: 10px;
            font-size: 1.5em;
        }
        .status {
            color: #565f89;
            font-size: 0.85em;
            margin-bottom: 20px;
        }
        .status .live {
            color: #9ece6a;
        }
        .status .cached {
            color: #f7768e;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
        .panel {
            background: #1a1b26;
            border: 1px solid #24283b;
            border-radius: 8px;
            padding: 15px;
        }
        .panel h2 {
            color: #7aa2f7;
            font-size: 1em;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #24283b;
        }
        .terrain-view {
            grid-column: 1 / -1;
        }
        .terrain-ascii {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.4;
            white-space: pre;
            color: #a9b1d6;
            background: #13141c;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .movement {
            color: #7dcfff;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .list-item {
            padding: 8px 0;
            border-bottom: 1px solid #24283b;
        }
        .list-item:last-child {
            border-bottom: none;
        }
        .list-item .label {
            color: #7aa2f7;
            font-size: 0.85em;
        }
        .list-item .value {
            color: #a9b1d6;
        }
        .presence-box {
            background: #1e1e2e;
            border-left: 3px solid #bb9af7;
            padding: 10px;
            margin-bottom: 10px;
        }
        .presence-name {
            color: #bb9af7;
            font-weight: bold;
        }
        .stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            color: #7aa2f7;
        }
        .stat-label {
            font-size: 0.8em;
            color: #565f89;
        }
        .question {
            color: #f7768e;
        }
        .quality-tag {
            display: inline-block;
            background: #24283b;
            color: #9ece6a;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 2px;
        }
        .relational {
            background: linear-gradient(135deg, #1a1b26 0%, #1e1e2e 100%);
            border-left: 3px solid #9ece6a;
        }
        .path-item {
            font-size: 0.9em;
            color: #565f89;
        }
        .error {
            color: #f7768e;
            padding: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>◉ ORE Terrain Viewer (Lite)</h1>
        <div class="status">
            <span id="live-indicator" class="live">● LIVE</span> | 
            Last update: <span id="last-update">-</span> |
            Session: <span id="session-id">-</span> |
            <span id="storage-path" style="font-size: 0.8em;"></span>
        </div>
        
        <div class="stats panel" style="margin-bottom: 20px;">
            <div class="stat">
                <div class="stat-value" id="stat-terrains">-</div>
                <div class="stat-label">Past Terrains</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-sessions">-</div>
                <div class="stat-label">All Sessions</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-questions">-</div>
                <div class="stat-label">Open Questions</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-presences">-</div>
                <div class="stat-label">Presences</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel terrain-view">
                <h2>TERRAIN</h2>
                <div id="presence-info"></div>
                <div class="terrain-ascii" id="terrain-ascii">Waiting for data...</div>
                <div class="movement" id="movement">-</div>
            </div>
            
            <div class="panel">
                <h2>OPEN QUESTIONS</h2>
                <div id="questions">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>PATH THIS SESSION</h2>
                <div id="path">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>KNOWN PRESENCES</h2>
                <div id="presences">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>PAST TERRAINS</h2>
                <div id="past-terrains">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/state')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('terrain-ascii').innerHTML = 
                            '<div class="error">Error: ' + data.error + '</div>';
                        return;
                    }
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    document.getElementById('session-id').textContent = data.session_id || '-';
                    document.getElementById('storage-path').textContent = data.storage_path || '';
                    
                    // Live status
                    const indicator = document.getElementById('live-indicator');
                    if (data.live) {
                        indicator.textContent = '● LIVE';
                        indicator.className = 'live';
                    } else {
                        indicator.textContent = '○ CACHED';
                        indicator.className = 'cached';
                    }
                    
                    // Stats
                    document.getElementById('stat-terrains').textContent = data.stats.past_terrains;
                    document.getElementById('stat-sessions').textContent = data.stats.all_sessions;
                    document.getElementById('stat-questions').textContent = data.stats.open_questions;
                    document.getElementById('stat-presences').textContent = data.stats.presences;
                    
                    // Presence info
                    if (data.current_presence) {
                        document.getElementById('presence-info').innerHTML = `
                            <div class="presence-box">
                                <span class="presence-name">✧ ${data.current_presence.name}</span>
                                <span style="color: #565f89"> | Session #${data.current_presence.sessions} | 
                                ${data.current_presence.threads} threads</span>
                            </div>
                        `;
                    } else {
                        document.getElementById('presence-info').innerHTML = '';
                    }
                    
                    // Terrain ASCII
                    document.getElementById('terrain-ascii').textContent = data.terrain_ascii;
                    
                    // Movement
                    document.getElementById('movement').textContent = 'Movement: ' + data.movement;
                    
                    // Questions
                    let qhtml = '';
                    if (!data.questions || data.questions.length === 0) {
                        qhtml = '<div style="color: #565f89">(no open questions)</div>';
                    } else {
                        data.questions.forEach(q => {
                            const shared = q.shared ? '<span style="color: #bb9af7">(shared)</span> ' : '';
                            qhtml += `<div class="list-item">
                                <span class="question">?</span> ${shared}${q.question}
                            </div>`;
                        });
                    }
                    document.getElementById('questions').innerHTML = qhtml;
                    
                    // Path
                    let phtml = '';
                    if (!data.path || data.path.length === 0) {
                        phtml = '<div style="color: #565f89">(no movement yet)</div>';
                    } else {
                        data.path.forEach(p => {
                            phtml += `<div class="path-item">${p.indicator} near ${p.landmark} (${p.distance.toFixed(2)})</div>`;
                        });
                    }
                    document.getElementById('path').innerHTML = phtml;
                    
                    // Presences
                    let preshtml = '';
                    if (!data.presences || data.presences.length === 0) {
                        preshtml = '<div style="color: #565f89">(no presences)</div>';
                    } else {
                        data.presences.forEach(p => {
                            preshtml += `<div class="list-item">
                                <div><strong>✧ ${p.name}</strong> [${p.hash}...]</div>
                                <div style="font-size: 0.85em; color: #565f89">
                                    Sessions: ${p.sessions} | Threads: ${p.threads}
                                </div>
                            </div>`;
                        });
                    }
                    document.getElementById('presences').innerHTML = preshtml;
                    
                    // Past terrains
                    let pthtml = '';
                    if (!data.past_terrains || data.past_terrains.length === 0) {
                        pthtml = '<div style="color: #565f89">(no past terrains)</div>';
                    } else {
                        data.past_terrains.slice(-5).forEach(t => {
                            const presence = t.presence_name ? ` (with ${t.presence_name})` : '';
                            pthtml += `<div class="list-item">
                                <div class="label">[${t.session_id}]${presence}</div>
                                <div class="value">${t.dominant_landmark} - ${t.thinking_about}</div>
                            </div>`;
                        });
                    }
                    document.getElementById('past-terrains').innerHTML = pthtml;
                })
                .catch(err => {
                    console.error('Update failed:', err);
                    document.getElementById('terrain-ascii').innerHTML = 
                        '<div class="error">Connection error: ' + err.message + '</div>';
                });
        }
        
        // Initial load
        updateDashboard();
        
        // Auto-refresh every 1 second
        setInterval(updateDashboard, 1000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    """Return current terrain state as JSON - reads directly from files."""
    try:
        live = get_live_state()
        
        # Load other data from files
        questions = load_json_file('questions.json')
        presences = load_json_file('presences.json')
        terrains = load_json_file('terrains.json')
        
        # Count sessions
        sessions_dir = TERRAIN_STORAGE / "sessions"
        session_count = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0
        
        # Build response
        data = {
            'session_id': live['session_id'] if live else 'unknown',
            'stats': {
                'past_terrains': len(terrains),
                'all_sessions': session_count,
                'open_questions': len([q for q in questions if not q.get('resolved', False)]),
                'presences': len(presences)
            },
            'terrain_ascii': '',
            'movement': live.get('movement_summary', 'no data') if live else 'no live data',
            'current_presence': None,
            'questions': [],
            'path': [],
            'presences': [],
            'past_terrains': [],
            'live': live is not None,
            'storage_path': str(TERRAIN_STORAGE)
        }
        
        # Render terrain from live positions
        if live and 'positions' in live:
            data['terrain_ascii'] = render_terrain_ascii(
                live['positions'], 
                live.get('current_presence')
            )
            
            # Current presence from live state
            if live.get('current_presence'):
                p = live['current_presence']
                data['current_presence'] = {
                    'name': p.get('name', 'Unknown'),
                    'hash': p.get('hash', '?')[:8],
                    'sessions': p.get('sessions', 0),
                    'threads': p.get('threads', 0)
                }
            
            # Path from live state
            prev_landmark = None
            for pos in live.get('position_history', [])[-15:]:
                distances = pos.get('distances', {})
                core = {k: v for k, v in distances.items() 
                       if not k.startswith(('unresolved:', 'shared-q:', 'relational:'))}
                if core:
                    closest = min(core.items(), key=lambda x: x[1])
                    indicator = '→→' if prev_landmark and prev_landmark != closest[0] else '··'
                    data['path'].append({
                        'landmark': closest[0],
                        'distance': closest[1],
                        'indicator': indicator
                    })
                    prev_landmark = closest[0]
        else:
            data['terrain_ascii'] = "Waiting for terrain data...\n\nRun terrain.py in another terminal and send a message."
        
        # Open questions
        for q in questions:
            if not q.get('resolved', False):
                data['questions'].append({
                    'id': q.get('id', '?')[:8],
                    'question': q.get('question', '')[:60] + ('...' if len(q.get('question', '')) > 60 else ''),
                    'shared': q.get('presence_hash') is not None
                })
        
        # Presences
        for p in presences:
            data['presences'].append({
                'name': p.get('name', 'Unknown'),
                'hash': p.get('hash', '?')[:8],
                'sessions': p.get('sessions_together', 0),
                'threads': len(p.get('open_threads', []))
            })
        
        # Past terrains
        for t in terrains[-7:]:
            presence_name = None
            if t.get('presence_hash'):
                for p in presences:
                    if p.get('hash') == t.get('presence_hash'):
                        presence_name = p.get('name')
                        break
            data['past_terrains'].append({
                'session_id': t.get('session_id', '?'),
                'dominant_landmark': t.get('dominant_landmark', '?'),
                'thinking_about': t.get('thinking_about', '')[:40] + '...',
                'presence_name': presence_name
            })
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'storage_path': str(TERRAIN_STORAGE)
        })


def main():
    import webbrowser
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Terrain Viewer (Lite)                                       ║
║  Standalone dashboard - no heavy dependencies                    ║
╚══════════════════════════════════════════════════════════════════╝

  Storage path: {TERRAIN_STORAGE}
  
  Starting server on http://localhost:5000
  Opening browser...
  
  Keep this running while using terrain.py in another terminal.
  Press Ctrl+C to stop.
""")
    
    # Open browser after short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:5000')
    
    Thread(target=open_browser, daemon=True).start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
