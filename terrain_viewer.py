#!/usr/bin/env python3
"""
ORE Terrain Viewer - Live Web Dashboard
========================================

Opens a browser window showing live terrain state.
Auto-refreshes when files change.

Usage:
    python terrain_viewer.py
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

# Add ore to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, render_template_string, jsonify
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

from terrain import UnifiedTerrain, TERRAIN_STORAGE
from canvas_reasoner import CanvasReasoner

app = Flask(__name__)

# Global terrain instance (reloaded on each request to get fresh data)
_last_load = 0
_terrain = None
_reasoner = None

def get_terrain():
    """Get terrain, preferring live state if available."""
    global _terrain, _reasoner, _last_load
    
    # Check for live state first (written on every message)
    live_path = TERRAIN_STORAGE / "live_state.json"
    
    # Check if any terrain files changed
    newest = 0
    for f in TERRAIN_STORAGE.glob("*.json"):
        mtime = f.stat().st_mtime
        if mtime > newest:
            newest = mtime
    
    # Also check sessions dir
    sessions_dir = TERRAIN_STORAGE / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("*.json"):
            mtime = f.stat().st_mtime
            if mtime > newest:
                newest = mtime
    
    # Reload if changed
    if newest > _last_load or _terrain is None:
        _reasoner = CanvasReasoner()
        _terrain = UnifiedTerrain(_reasoner)
        _last_load = time.time()
    
    return _terrain


def get_live_state():
    """Get live state directly from file if available."""
    live_path = TERRAIN_STORAGE / "live_state.json"
    if live_path.exists():
        try:
            with open(live_path) as f:
                return json.load(f)
        except:
            pass
    return None

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
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
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
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            line-height: 1.4;
            white-space: pre;
            color: #a9b1d6;
            background: #13141c;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .presence-active {
            color: #bb9af7;
        }
        .landmark-close {
            color: #9ece6a;
            font-weight: bold;
        }
        .landmark-mid {
            color: #e0af68;
        }
        .landmark-far {
            color: #565f89;
        }
        .question {
            color: #f7768e;
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
        .history-item {
            font-size: 0.9em;
            padding: 5px 0;
            color: #565f89;
        }
        .history-item .time {
            color: #7aa2f7;
        }
        .relational {
            background: linear-gradient(135deg, #1a1b26 0%, #1e1e2e 100%);
            border-left: 3px solid #9ece6a;
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
        .breakthrough {
            color: #e0af68;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>◉ ORE Terrain Viewer</h1>
        <div class="status">
            <span class="live" id="live-indicator">● LIVE</span> | 
            Last update: <span id="last-update">-</span> |
            Session: <span id="session-id">-</span> |
            <span id="storage-path" style="color: #565f89; font-size: 0.8em;"></span>
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
            <div class="stat">
                <div class="stat-value" id="stat-relational">-</div>
                <div class="stat-label">Relational Landmarks</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel terrain-view">
                <h2>TERRAIN</h2>
                <div id="presence-info"></div>
                <div class="terrain-ascii" id="terrain-ascii">Loading...</div>
                <div class="movement" id="movement">-</div>
            </div>
            
            <div class="panel">
                <h2>OPEN QUESTIONS</h2>
                <div id="questions">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>RELATIONAL LANDMARKS</h2>
                <div id="relational">Loading...</div>
            </div>
            
            <div class="panel">
                <h2>PATH THIS SESSION</h2>
                <div id="path">Loading...</div>
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
                    // Update timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                    document.getElementById('session-id').textContent = data.session_id || '-';
                    
                    // Storage path and live status
                    const liveStatus = data.live ? '● LIVE' : '○ CACHED';
                    document.getElementById('live-indicator').textContent = liveStatus;
                    document.getElementById('live-indicator').style.color = data.live ? '#9ece6a' : '#f7768e';
                    document.getElementById('storage-path').textContent = data.storage_path || '';
                    
                    // Stats
                    document.getElementById('stat-terrains').textContent = data.stats.past_terrains;
                    document.getElementById('stat-sessions').textContent = data.stats.all_sessions;
                    document.getElementById('stat-questions').textContent = data.stats.open_questions;
                    document.getElementById('stat-presences').textContent = data.stats.presences;
                    document.getElementById('stat-relational').textContent = data.stats.relational_landmarks;
                    
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
                    if (data.questions.length === 0) {
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
                    
                    // Relational landmarks
                    let rlhtml = '';
                    if (data.relational_landmarks.length === 0) {
                        rlhtml = '<div style="color: #565f89">(none yet)</div>';
                    } else {
                        data.relational_landmarks.forEach(rl => {
                            let qualities = rl.qualities.map(q => `<span class="quality-tag">${q}</span>`).join('');
                            let breakthroughs = rl.breakthroughs.slice(-2).map(b => 
                                `<div class="breakthrough">• ${b}</div>`
                            ).join('');
                            rlhtml += `<div class="list-item relational">
                                <div><strong>${rl.symbol} ${rl.name}</strong></div>
                                <div style="font-size: 0.85em; color: #565f89">
                                    With: ${rl.presence_name} | Visited: ${rl.times_visited}x
                                </div>
                                ${qualities ? '<div style="margin-top: 5px">' + qualities + '</div>' : ''}
                                ${breakthroughs}
                            </div>`;
                        });
                    }
                    document.getElementById('relational').innerHTML = rlhtml;
                    
                    // Path
                    let phtml = '';
                    if (data.path.length === 0) {
                        phtml = '<div style="color: #565f89">(no movement yet)</div>';
                    } else {
                        data.path.forEach(p => {
                            phtml += `<div class="history-item">${p.indicator} near ${p.landmark} (${p.distance.toFixed(2)})</div>`;
                        });
                    }
                    document.getElementById('path').innerHTML = phtml;
                    
                    // Past terrains
                    let pthtml = '';
                    if (data.past_terrains.length === 0) {
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
                .catch(err => console.error('Update failed:', err));
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
    """Return current terrain state as JSON."""
    # Force fresh load every time for live updates
    global _terrain, _reasoner, _last_load
    _reasoner = CanvasReasoner()
    _terrain = UnifiedTerrain(_reasoner)
    _last_load = time.time()
    
    terrain = _terrain
    live = get_live_state()
    
    # Build response - use live state for real-time data where available
    data = {
        'session_id': live['session_id'] if live else terrain.session_id,
        'stats': {
            'past_terrains': len(terrain.past_terrains),
            'all_sessions': len(terrain.all_sessions),
            'open_questions': len(terrain.get_open_questions()),
            'presences': len(terrain.presences),
            'relational_landmarks': len(terrain.relational_landmarks)
        },
        'terrain_ascii': terrain.render_first_person(),
        'movement': live['movement_summary'] if live else terrain.get_movement_summary(),
        'current_presence': None,
        'current_relational': None,
        'questions': [],
        'relational_landmarks': [],
        'path': [],
        'past_terrains': [],
        'live': live is not None,
        'storage_path': str(TERRAIN_STORAGE),
        'live_file_exists': (TERRAIN_STORAGE / "live_state.json").exists()
    }
    
    # Use live positions if available
    if live and 'positions' in live:
        # Update terrain landmarks with live positions for accurate rendering
        for name, pos_data in live['positions'].items():
            if name in terrain.landmarks:
                terrain.landmarks[name].current_distance = pos_data['distance']
                terrain.landmarks[name].velocity = pos_data.get('velocity', 0)
                terrain.landmarks[name].resonance = pos_data.get('resonance', 0)
        # Re-render with live positions
        data['terrain_ascii'] = terrain.render_first_person()
    
    # Current presence - prefer live
    if live and live.get('current_presence'):
        p = live['current_presence']
        data['current_presence'] = {
            'name': p['name'],
            'hash': p['hash'][:8],
            'sessions': p['sessions'],
            'threads': p['threads'],
            'signature': p.get('signature', [])[:3],
            'grooves': p.get('grooves', [])[:3]
        }
    elif terrain.current_presence:
        p = terrain.current_presence
        data['current_presence'] = {
            'name': p.name,
            'hash': p.hash[:8],
            'sessions': p.sessions_together,
            'threads': len(p.open_threads),
            'signature': p.thinking_signature[:3],
            'grooves': p.shared_grooves[:3]
        }
    
    # Current relational landmark - from live
    if live and live.get('current_relational'):
        rl = live['current_relational']
        data['current_relational'] = {
            'name': rl['name'],
            'qualities': rl.get('qualities', []),
            'breakthroughs': rl.get('breakthroughs', []),
            'times_visited': rl.get('times_visited', 0)
        }
    
    # Questions
    for q in terrain.get_open_questions()[:10]:
        data['questions'].append({
            'id': q.id[:8],
            'question': q.question[:60] + ('...' if len(q.question) > 60 else ''),
            'shared': q.presence_hash is not None,
            'pull': q.pull_strength
        })
    
    # Relational landmarks
    for rl in terrain.relational_landmarks.values():
        presence_name = terrain.presences[rl.presence_hash].name if rl.presence_hash in terrain.presences else 'unknown'
        data['relational_landmarks'].append({
            'id': rl.id,
            'name': rl.name,
            'symbol': rl.get_symbol(),
            'presence_name': presence_name,
            'qualities': rl.qualities[:4],
            'times_visited': rl.times_visited,
            'breakthroughs': [b[:50] + '...' if len(b) > 50 else b for b in rl.breakthroughs[-3:]]
        })
    
    # Path - use live position history if available
    path_data = live.get('position_history', []) if live else terrain.position_history
    prev_landmark = None
    for pos in path_data[-15:]:
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
    
    # Past terrains
    for t in terrain.past_terrains[-7:]:
        presence_name = None
        if t.presence_hash and t.presence_hash in terrain.presences:
            presence_name = terrain.presences[t.presence_hash].name
        data['past_terrains'].append({
            'session_id': t.session_id,
            'dominant_landmark': t.dominant_landmark,
            'thinking_about': t.thinking_about[:40] + '...',
            'presence_name': presence_name
        })
    
    return jsonify(data)


def main():
    import webbrowser
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  ORE Terrain Viewer                                              ║
║  Live web dashboard for cognitive terrain                        ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    print("  Starting server on http://localhost:5000")
    print("  Opening browser...")
    print("")
    print("  Keep this running while using terrain.py in another terminal.")
    print("  Press Ctrl+C to stop.")
    print("")
    
    # Open browser after short delay
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:5000')
    
    Thread(target=open_browser, daemon=True).start()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
