"""
AeroWatch AI — Autonomous AI Interceptor Missile & Fire Control Simulator
========================================================================
Interactive tactical simulation featuring:
- Autonomous AI Proportional Navigation (PN) & Augmented PN (APN) guidance.
- Dynamic TVC (Thrust Vector Control) snap-alignment: impossible for target to evade, even with high-G breaks or manual user dragging!
- Real warplane, jet fighter, and combat drone aerodynamic models & silhouettes.
- Terminal kinetic detonation with directional blast fragmentation particle dynamics.
- Web Audio tactical sound synthesizer (rocket roar, accelerating seeker lock tones, blast detonation).
- Spatial visionOS frosted quartz glass aesthetic.
"""

from typing import Optional


def get_interceptor_sim_html(
    target_type: str = "Warplane",
    evasion_mode: str = "High-G Weave",
    evasion_g: float = 6.0,
    nav_gain: float = 4.5,
    sound_enabled: bool = True,
) -> str:
    """Generates the self-contained 60 FPS HTML5/Canvas simulation widget."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
    -webkit-user-select: none;
  }}
  body {{
    background: transparent;
    color: #FFFFFF;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif;
    overflow: hidden;
  }}

  /* Outer Glass Panel Container */
  .interceptor-panel {{
    background: rgba(22, 22, 27, 0.72);
    backdrop-filter: blur(50px) saturate(190%);
    -webkit-backdrop-filter: blur(50px) saturate(190%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-top: 1.8px solid rgba(255, 255, 255, 0.40);
    border-left: 1.2px solid rgba(255, 255, 255, 0.20);
    border-radius: 28px;
    padding: 22px;
    box-shadow: 
      0 32px 75px rgba(0, 0, 0, 0.65),
      inset 0 1.5px 2px rgba(255, 255, 255, 0.35),
      inset 0 -1px 2px rgba(0, 0, 0, 0.45);
    max-width: 100%;
    margin: 0 auto;
  }}

  /* Top Tactical HUD Bar */
  .hud-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }}
  .hud-title-box {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .pulse-emitter {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #EF4444;
    box-shadow: 0 0 12px #EF4444;
    animation: radarPulse 1.4s infinite ease-out;
  }}
  @keyframes radarPulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }}
    70% {{ box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
  }}
  .hud-title {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #F8FAFC;
    text-transform: uppercase;
  }}
  .hud-sub {{
    font-size: 10.5px;
    color: #94A3B8;
    letter-spacing: 0.06em;
  }}

  /* Telemetry Chips */
  .telemetry-strip {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .telem-chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    background: rgba(0, 0, 0, 0.38);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-top: 1.2px solid rgba(255, 255, 255, 0.28);
    border-radius: 9999px;
    padding: 4px 12px;
    color: #C7C7CC;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.35);
  }}
  .telem-chip span.val {{
    color: #FFFFFF;
    font-weight: 700;
  }}
  .telem-chip.active {{
    border-color: rgba(56, 189, 248, 0.4);
    border-top-color: rgba(56, 189, 248, 0.8);
    color: #38BDF8;
  }}

  /* Main Radar Viewport Layout */
  .viewport-wrapper {{
    position: relative;
    width: 100%;
    height: 480px;
    border-radius: 20px;
    overflow: hidden;
    background: radial-gradient(ellipse at 50% 50%, #15151B 0%, #0A0A0E 100%);
    border: 1.5px solid rgba(255, 255, 255, 0.12);
    box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.85);
  }}
  #radarCanvas {{
    display: block;
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }}

  /* Overlay HUD Crosshairs & Status */
  .hud-overlay-status {{
    position: absolute;
    top: 16px;
    left: 18px;
    pointer-events: none;
    font-family: 'JetBrains Mono', monospace;
  }}
  .hud-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(239, 68, 68, 0.18);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
    margin-bottom: 6px;
  }}
  .hud-target-name {{
    font-size: 13px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 0.06em;
  }}
  .hud-target-stats {{
    font-size: 10.5px;
    color: #94A3B8;
    margin-top: 3px;
    line-height: 1.4;
  }}

  /* HUD Live Intercept Callout */
  .hud-intercept-callout {{
    position: absolute;
    top: 16px;
    right: 18px;
    pointer-events: none;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
  }}
  .tti-big {{
    font-size: 24px;
    font-weight: 800;
    color: #38BDF8;
    text-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
  }}
  .tti-label {{
    font-size: 9.5px;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.10em;
    text-transform: uppercase;
  }}

  /* Bottom Controls Toolbar */
  .controls-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 18px;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .action-btn {{
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.45) 0%, rgba(220, 38, 38, 0.25) 100%);
    border: 1px solid rgba(239, 68, 68, 0.6);
    border-top: 1.5px solid rgba(255, 255, 255, 0.6);
    color: #FFFFFF;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 11px 24px;
    border-radius: 9999px;
    cursor: pointer;
    box-shadow: 0 8px 24px rgba(220, 38, 38, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.4);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .action-btn:hover {{
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.65) 0%, rgba(220, 38, 38, 0.40) 100%);
    box-shadow: 0 12px 32px rgba(220, 38, 38, 0.55), inset 0 1.5px 2px #FFFFFF;
    transform: translateY(-2px) scale(1.02);
  }}
  .action-btn:active {{
    transform: translateY(1px) scale(0.98);
  }}
  .action-btn.reset-btn {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-top: 1.2px solid rgba(255, 255, 255, 0.35);
    color: #E2E8F0;
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.2);
  }}
  .action-btn.reset-btn:hover {{
    background: rgba(255, 255, 255, 0.16);
    color: #FFFFFF;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
  }}

  /* Mode Selectors in Controls */
  .select-group {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .custom-select {{
    background: rgba(0, 0, 0, 0.40);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-top: 1.2px solid rgba(255, 255, 255, 0.30);
    border-radius: 12px;
    color: #FFFFFF;
    padding: 7px 12px;
    font-size: 11.5px;
    font-weight: 600;
    outline: none;
    cursor: pointer;
  }}
  .custom-select option {{
    background: #18171C;
    color: #FFFFFF;
  }}

  /* Audio & Drag Hint */
  .hint-text {{
    font-size: 11px;
    color: #38BDF8;
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
  }}
  .hint-icon {{
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #38BDF8;
    box-shadow: 0 0 8px #38BDF8;
  }}
</style>
</head>
<body>

<div class="interceptor-panel">
  <!-- Top HUD Bar -->
  <div class="hud-topbar">
    <div class="hud-title-box">
      <div class="pulse-emitter" id="threatPulse"></div>
      <div>
        <div class="hud-title">AIRSPACE KINETIC FIRE CONTROL // C2 INTERCEPTOR</div>
        <div class="hud-sub">Autonomous TVC & APN Guided Missile with Guaranteed Evasion Intercept</div>
      </div>
    </div>
    <div class="telemetry-strip">
      <div class="telem-chip" id="chipStatus">STATUS: <span class="val" id="valStatus">STANDBY</span></div>
      <div class="telem-chip active">GUIDANCE: <span class="val">TVC + APN HYBRID</span></div>
      <div class="telem-chip">SPEED: <span class="val" id="valMach">Mach 4.2</span></div>
      <div class="telem-chip">KINETIC ENERGY: <span class="val" id="valKe">18.5 MJ</span></div>
      <div class="telem-chip">KILL PROBABILITY: <span class="val" id="valPk" style="color: #34D399;">100.0% Pk</span></div>
      <div class="telem-chip">CLOSING VELOCITY: <span class="val" id="valVc">-- m/s</span></div>
      <div class="telem-chip">TVC PULL: <span class="val" id="valG">0.0 G</span></div>
    </div>
  </div>

  <!-- Radar Canvas Viewport -->
  <div class="viewport-wrapper">
    <canvas id="radarCanvas"></canvas>

    <!-- Overlay Status Top-Left -->
    <div class="hud-overlay-status">
      <div class="hud-badge" id="hudBadge">HOSTILE AIRSPACE INTRUDER</div>
      <div class="hud-target-name" id="hudTargetName">TARGET: 1x SUPERSONIC WARPLANE</div>
      <div class="hud-target-stats" id="hudTargetStats">
        RANGE: 1,650 m | BEARING: 042° | ALT: 850 m<br>
        SPEED: 180 m/s | EVASION VECTOR: ACTIVE
      </div>
    </div>

    <!-- Overlay TTI Top-Right -->
    <div class="hud-intercept-callout">
      <div class="tti-label">ESTIMATED TIME TO IMPACT</div>
      <div class="tti-big" id="hudTTI">--.- s</div>
      <div style="font-size: 10px; color: #94A3B8; margin-top: 2px;">SLANT RANGE: <span id="hudSlantRange" style="color: #FFFFFF; font-weight: 700;">1,650 m</span></div>
    </div>
  </div>

  <!-- Bottom Interactive Controls Bar -->
  <div class="controls-bar">
    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
      <button class="action-btn" id="btnLaunch">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <polygon points="12 2 15 8 21 9 17 14 18 20 12 17 6 20 7 14 3 9 9 8 12 2"></polygon>
        </svg>
        LOCK & FIRE INTERCEPTOR
      </button>

      <button class="action-btn reset-btn" id="btnReset">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 0 1 15-6.7L21 8M21 3v5h-5M21 12a9 9 0 0 1-15 6.7L3 16M3 21v-5h5"></path>
        </svg>
        RESET
      </button>

      <div class="select-group">
        <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">ROCKET WEAPON POWER:</span>
        <select class="custom-select" id="selectMissileMode">
          <option value="salvo">🚀 Dual-Missile Salvo Battery (Alpha + Bravo Pincer • 100% Pk)</option>
          <option value="hypersonic">⚡ Hypersonic Sprint Interceptor (Mach 4.5 • 48G TVC • 18.5 MJ)</option>
          <option value="standard">🎯 Tactical Interceptor (Mach 3.0 • 35G)</option>
        </select>
      </div>

      <div class="select-group">
        <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">TARGET THREAT:</span>
        <select class="custom-select" id="selectTargetType">
          <option value="Warplane">Supersonic Warplane / Strike Jet (Mach 1.9)</option>
          <option value="Kamikaze">Low-Altitude Kamikaze Attack Drone</option>
          <option value="Drone">High-Agility Recon Quadcopter (FPV)</option>
          <option value="JetFighter">High-G Combat Fighter Jet (Evasive Break)</option>
        </select>
      </div>

      <div class="select-group">
        <span style="font-size: 11px; color: #94A3B8; font-weight: 600;">EVASION TACTIC:</span>
        <select class="custom-select" id="selectEvasion">
          <option value="weave">High-G Oscillating S-Turns (6.5G Weave)</option>
          <option value="dive">Sudden 90° Break Dive & Climb</option>
          <option value="corkscrew">Erratic Multi-Axis Corkscrew</option>
          <option value="dash">Maximum Afterburner Emergency Dash</option>
          <option value="drag">🕹️ Interactive User Mouse Flight</option>
        </select>
      </div>
    </div>

    <div style="display: flex; align-items: center; gap: 14px;">
      <div class="hint-text">
        <span class="hint-icon"></span>
        <span>Drag target or click anywhere — Rocket will hunt & destroy it!</span>
      </div>
      <button class="action-btn reset-btn" id="btnAudioToggle" style="padding: 7px 14px; font-size: 11px;">
        🔊 AUDIO: ON
      </button>
    </div>
  </div>
</div>

<script>
// =============================================================================
// WEB AUDIO SYNTHESIZER
// =============================================================================
let audioCtx = null;
let soundEnabled = {str(sound_enabled).lower()};

function getAudioContext() {{
  if (!audioCtx) {{
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) audioCtx = new AudioContext();
  }}
  if (audioCtx && audioCtx.state === 'suspended') {{
    audioCtx.resume();
  }}
  return audioCtx;
}}

function playLaunchSound() {{
  if (!soundEnabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;
  try {{
    const bufferSize = ctx.sampleRate * 1.5;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {{
      data[i] = Math.random() * 2 - 1;
    }}
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(350, ctx.currentTime);
    filter.frequency.exponentialRampToValueAtTime(850, ctx.currentTime + 1.2);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.01, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.5, ctx.currentTime + 0.15);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 1.5);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    noise.start();
  }} catch (e) {{}}
}}

let lastBeepTime = 0;
function playSeekerBeep(rateHz) {{
  if (!soundEnabled) return;
  const now = performance.now();
  const interval = 1000 / Math.max(1, Math.min(18, rateHz));
  if (now - lastBeepTime < interval) return;
  lastBeepTime = now;

  const ctx = getAudioContext();
  if (!ctx) return;
  try {{
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1600, ctx.currentTime);
    gain.gain.setValueAtTime(0.07, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.035);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.04);
  }} catch (e) {{}}
}}

function playExplosionSound() {{
  if (!soundEnabled) return;
  const ctx = getAudioContext();
  if (!ctx) return;
  try {{
    const bufferSize = ctx.sampleRate * 2.0;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {{
      data[i] = (Math.random() * 2 - 1) * Math.exp(-2.5 * i / bufferSize);
    }}
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(260, ctx.currentTime);
    filter.frequency.linearRampToValueAtTime(60, ctx.currentTime + 1.4);

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.9, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 2.0);

    noise.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);
    noise.start();
  }} catch (e) {{}}
}}


// =============================================================================
// SIMULATION ENGINE: UNBEATABLE TVC + APN HYBRID MISSILE
// =============================================================================
const canvas = document.getElementById('radarCanvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {{
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
}}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Target State
let target = {{
  type: '{target_type}',
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  speed: 150,
  heading: 0,
  altitude: 850,
  evasionMode: 'weave',
  evasionG: {evasion_g},
  trail: [],
  destroyed: false,
  destroyedTimer: 0
}};

// Missile State (Alpha - Primary Interceptor)
let missile = {{
  active: false,
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  speed: 0,
  baseSpeed: 580,
  accel: 850,
  heading: -Math.PI / 2,
  stage: 'STANDBY',
  flightTime: 0,
  trail: [],
  smokeParticles: []
}};

// Missile Bravo (Flanker Interceptor for Dual-Salvo Battery)
let missileB = {{
  active: false,
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  speed: 0,
  baseSpeed: 620,
  accel: 950,
  heading: -Math.PI / 2 - 0.20,
  stage: 'STANDBY',
  flightTime: 0,
  trail: [],
  smokeParticles: []
}};

let explosionParticles = [];
let debrisParticles = [];
let radarSweepAngle = 0;

function getLauncherPos() {{
  return {{ x: canvas.width / 2, y: canvas.height - 35 }};
}}

function initScenario() {{
  const lw = canvas.width;
  const lh = canvas.height;
  
  target.type = document.getElementById('selectTargetType').value;
  target.evasionMode = document.getElementById('selectEvasion').value;
  target.x = lw * 0.20;
  target.y = lh * 0.24;
  target.heading = Math.PI * 0.06;

  // Set realistic speed based on threat type
  if (target.type === 'Warplane' || target.type === 'JetFighter') {{
    target.speed = 180;
    target.altitude = 1200;
  }} else if (target.type === 'Kamikaze') {{
    target.speed = 140;
    target.altitude = 350;
  }} else {{
    target.speed = 95;
    target.altitude = 180;
  }}

  target.vx = Math.cos(target.heading) * target.speed;
  target.vy = Math.sin(target.heading) * target.speed;
  target.trail = [];
  target.destroyed = false;
  target.destroyedTimer = 0;

  const lp = getLauncherPos();
  missile.active = false;
  missile.x = lp.x;
  missile.y = lp.y;
  missile.vx = 0;
  missile.vy = 0;
  missile.speed = 0;
  missile.heading = -Math.PI / 2;
  missile.stage = 'STANDBY';
  missile.flightTime = 0;
  missile.trail = [];
  missile.smokeParticles = [];

  missileB.active = false;
  missileB.x = lp.x + 14;
  missileB.y = lp.y;
  missileB.vx = 0;
  missileB.vy = 0;
  missileB.speed = 0;
  missileB.heading = -Math.PI / 2 - 0.20;
  missileB.stage = 'STANDBY';
  missileB.flightTime = 0;
  missileB.trail = [];
  missileB.smokeParticles = [];

  explosionParticles = [];
  debrisParticles = [];

  document.getElementById('valStatus').innerText = 'STANDBY // READY';
  document.getElementById('valStatus').style.color = '#FFFFFF';
  document.getElementById('chipStatus').className = 'telem-chip';
  document.getElementById('valMach').innerText = 'Mach 4.2';
  document.getElementById('valKe').innerText = '18.5 MJ';
  document.getElementById('valPk').innerText = '100.0% Pk';
  document.getElementById('valVc').innerText = '-- m/s';
  document.getElementById('valG').innerText = '0.0 G';
  document.getElementById('hudTTI').innerText = '--.- s';
  document.getElementById('hudTTI').style.color = '#38BDF8';
  document.getElementById('hudSlantRange').innerText = '1,650 m';
  document.getElementById('hudBadge').innerText = 'HOSTILE AIRSPACE INTRUDER';
  document.getElementById('hudBadge').style.color = '#F87171';
  document.getElementById('hudBadge').style.borderColor = 'rgba(239, 68, 68, 0.4)';
  document.getElementById('hudBadge').style.background = 'rgba(239, 68, 68, 0.18)';
  document.getElementById('threatPulse').style.background = '#EF4444';
  document.getElementById('threatPulse').style.boxShadow = '0 0 12px #EF4444';

  let tName = 'SUPERSONIC WARPLANE';
  if (target.type === 'Kamikaze') tName = 'KAMIKAZE ATTACK DRONE';
  else if (target.type === 'Drone') tName = 'RECON FPV QUADCOPTER';
  else if (target.type === 'JetFighter') tName = 'COMBAT FIGHTER JET';
  document.getElementById('hudTargetName').innerText = `TARGET: 1x ${{tName}}`;
}}

function launchInterceptor() {{
  if (missile.active || target.destroyed) return;
  getAudioContext();
  playLaunchSound();

  const lp = getLauncherPos();
  const mode = document.getElementById('selectMissileMode').value;

  missile.active = true;
  missile.x = lp.x - (mode === 'salvo' ? 14 : 0);
  missile.y = lp.y;
  missile.heading = -Math.PI / 2;
  missile.speed = mode === 'hypersonic' ? 360 : 240;
  missile.accel = mode === 'hypersonic' ? 1200 : 850;
  missile.vx = Math.cos(missile.heading) * missile.speed;
  missile.vy = Math.sin(missile.heading) * missile.speed;
  missile.stage = 'BOOST MOTOR ACTIVE';
  missile.flightTime = 0;
  missile.trail = [];

  if (mode === 'salvo') {{
    missileB.active = true;
    missileB.x = lp.x + 14;
    missileB.y = lp.y;
    missileB.heading = -Math.PI / 2 - 0.22;
    missileB.speed = 260;
    missileB.accel = 950;
    missileB.vx = Math.cos(missileB.heading) * missileB.speed;
    missileB.vy = Math.sin(missileB.heading) * missileB.speed;
    missileB.stage = 'SALVO FLANKER BOOST';
    missileB.flightTime = 0;
    missileB.trail = [];
  }} else {{
    missileB.active = false;
  }}

  document.getElementById('valStatus').innerText = mode === 'salvo' ? 'DUAL SALVO // TVC LOCK' : 'BOOST // TVC LOCK';
  document.getElementById('chipStatus').className = 'telem-chip active';
}}

// Interactive Drag / Click anywhere
let isDraggingTarget = false;
let lastMouseTime = performance.now();

function setTargetToPos(mx, my) {{
  const now = performance.now();
  const timeDiff = Math.max(0.005, (now - lastMouseTime) / 1000);
  const dx = mx - target.x;
  const dy = my - target.y;
  const dist = Math.hypot(dx, dy);
  if (dist > 1) {{
    target.heading = Math.atan2(dy, dx);
    target.vx = Math.max(-450, Math.min(450, dx / timeDiff));
    target.vy = Math.max(-450, Math.min(450, dy / timeDiff));
  }}
  target.x = mx;
  target.y = my;
  lastMouseTime = now;

  // Auto-rearm threat if user clicks or drags after previous destruction
  if (target.destroyed) {{
    target.destroyed = false;
    target.destroyedTimer = 0;
    explosionParticles = [];
    debrisParticles = [];
    document.getElementById('hudBadge').innerText = 'HOSTILE AIRSPACE INTRUDER';
    document.getElementById('hudBadge').style.color = '#F87171';
    document.getElementById('threatPulse').style.background = '#EF4444';
  }}
}}

canvas.addEventListener('mousedown', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  isDraggingTarget = true;
  lastMouseTime = performance.now();
  setTargetToPos(mx, my);
}});

canvas.addEventListener('mousemove', (e) => {{
  if (!isDraggingTarget) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  setTargetToPos(mx, my);
}});

window.addEventListener('mouseup', () => {{ isDraggingTarget = false; }});


// =============================================================================
// MAIN 60 FPS SIMULATION LOOP
// =============================================================================
let lastTimestamp = performance.now();

function update(dt) {{
  radarSweepAngle += dt * 1.6;

  // 1. UPDATE TARGET (If not manually dragged)
  if (!target.destroyed) {{
    const lw = canvas.width;
    const lh = canvas.height;

    if (!isDraggingTarget) {{
      if (target.evasionMode === 'weave') {{
        // Sinusoidal oscillating weave (6.5G)
        const omega = 4.2;
        const lateralAccel = Math.sin(performance.now() * 0.0035 * omega) * (target.evasionG * 40);
        target.heading += (lateralAccel / target.speed) * dt;
      }} else if (target.evasionMode === 'dive') {{
        // Sudden 90° Break Dive if missile gets within 350px
        if (missile.active) {{
          const dist = Math.hypot(missile.x - target.x, missile.y - target.y);
          if (dist < 350) {{
            target.heading += dt * 5.5; // Radical break
            target.speed = Math.min(260, target.speed + dt * 120);
          }}
        }}
      }} else if (target.evasionMode === 'corkscrew') {{
        target.heading += (Math.sin(performance.now() * 0.005) * 3.5 + Math.cos(performance.now() * 0.008) * 2.0) * dt;
      }} else if (target.evasionMode === 'dash') {{
        target.speed = Math.min(300, target.speed + dt * 100);
      }}

      target.vx = Math.cos(target.heading) * target.speed;
      target.vy = Math.sin(target.heading) * target.speed;

      target.x += target.vx * dt;
      target.y += target.vy * dt;

      // Soft boundary reflection
      if (target.x < 50) {{ target.x = 50; target.heading = Math.PI - target.heading; }}
      if (target.x > lw - 50) {{ target.x = lw - 50; target.heading = Math.PI - target.heading; }}
      if (target.y < 40) {{ target.y = 40; target.heading = -target.heading; }}
      if (target.y > lh * 0.70) {{ target.y = lh * 0.70; target.heading = -target.heading; }}
    }}

    target.trail.push({{ x: target.x, y: target.y }});
    if (target.trail.length > 40) target.trail.shift();
  }}

  // 2. UPDATE MISSILES: UNBEATABLE PIF-PAF TVC + APN DUAL SALVO ENGINE
  const activeMissiles = [];
  if (missile.active && !target.destroyed) activeMissiles.push({{ m: missile, isFlanker: false, name: 'ALPHA' }});
  if (missileB.active && !target.destroyed) activeMissiles.push({{ m: missileB, isFlanker: true, name: 'BRAVO' }});

  for (const item of activeMissiles) {{
    const curM = item.m;
    curM.flightTime += dt;

    const dx = target.x - curM.x;
    const dy = target.y - curM.y;
    const R = Math.hypot(dx, dy);

    // Dynamic speed dominance: Rocket is ALWAYS faster than the target!
    const targetCurrentSpeed = Math.hypot(target.vx, target.vy);
    const requiredSpeed = Math.max(580, Math.min(1000, targetCurrentSpeed * 2.3));
    if (curM.speed < requiredSpeed) {{
      curM.speed = Math.min(requiredSpeed, curM.speed + curM.accel * dt * 2.5);
    }}

    // Lead Point Calculation (Predictive APN Intercept Point)
    const ttiEst = Math.max(0.01, R / Math.max(120, curM.speed));
    const leadFactor = isDraggingTarget ? 0.15 : Math.min(0.75, ttiEst);
    const leadX = target.x + target.vx * leadFactor;
    const leadY = target.y + target.vy * leadFactor;
    
    // Flanker Bravo applies an offset angle to close the pincer trap
    const flankOffset = (item.isFlanker && R > 130) ? 0.22 : 0.0;
    const desiredHeading = Math.atan2(leadY - curM.y, leadX - curM.x) + flankOffset;

    // Normalized heading error to lead target
    let headingError = desiredHeading - curM.heading;
    while (headingError > Math.PI) headingError -= Math.PI * 2;
    while (headingError < -Math.PI) headingError += Math.PI * 2;

    // Closing velocity
    const vRelX = target.vx - curM.vx;
    const vRelY = target.vy - curM.vy;
    const Vc = -(dx * vRelX + dy * vRelY) / Math.max(1, R);

    // High-Authority Hypersonic TVC snap turning: up to 42 rad/s (>2,400 deg/s)
    const maxTurnRate = (Math.abs(headingError) > 0.35 || isDraggingTarget) ? 42.0 : 26.0;
    const turnStep = Math.sign(headingError) * Math.min(maxTurnRate, Math.abs(headingError) * 24.0) * dt;

    if (Math.abs(headingError) <= Math.abs(turnStep)) {{
      curM.heading = desiredHeading;
    }} else {{
      curM.heading += turnStep;
    }}

    // Lateral Divert Thruster assistance (instantly vector velocities)
    const targetVx = Math.cos(curM.heading) * curM.speed;
    const targetVy = Math.sin(curM.heading) * curM.speed;
    const divertBlend = Math.min(1.0, dt * 26.0);
    curM.vx = curM.vx * (1 - divertBlend) + targetVx * divertBlend;
    curM.vy = curM.vy * (1 - divertBlend) + targetVy * divertBlend;

    // Save previous position for continuous collision sweep
    const prevMx = curM.x;
    const prevMy = curM.y;

    curM.x += curM.vx * dt;
    curM.y += curM.vy * dt;

    // Stage designation
    if (isDraggingTarget) {{
      curM.stage = 'PILOT EVASION COUNTERED // TVC PURSUIT';
    }} else if (R < 150) {{
      curM.stage = 'TERMINAL KINETIC HOMING';
    }} else {{
      curM.stage = item.isFlanker ? 'SALVO BRAVO PINCER LOCK' : 'APN HYPERSONIC LOCK';
    }}

    // Calculate live G-load pulled
    const liveG = Math.abs(turnStep / Math.max(0.001, dt) * curM.speed) / (9.81 * 8.0);
    document.getElementById('valG').innerText = `${{Math.min(95.0, liveG).toFixed(1)}} G`;

    // Continuous Line-Segment Collision Check: prevents any tunneling!
    const segDx = curM.x - prevMx;
    const segDy = curM.y - prevMy;
    const segLenSq = segDx * segDx + segDy * segDy;
    let tParam = segLenSq === 0 ? 0 : ((target.x - prevMx) * segDx + (target.y - prevMy) * segDy) / segLenSq;
    tParam = Math.max(0, Math.min(1, tParam));
    const closestX = prevMx + tParam * segDx;
    const closestY = prevMy + tParam * segDy;
    const missDist = Math.hypot(target.x - closestX, target.y - closestY);

    // Telemetry updates (driven by primary or closest missile)
    if (!item.isFlanker || !missile.active) {{
      document.getElementById('valStatus').innerText = curM.stage;
      document.getElementById('valVc').innerText = `${{Math.round(Math.max(0, Vc) * 2.5)}} m/s`;
      const tti = Math.max(0, R / Math.max(40, curM.speed));
      document.getElementById('hudTTI').innerText = `${{tti.toFixed(1)}} s`;
      document.getElementById('hudSlantRange').innerText = `${{Math.round(R * 3.5)}} m`;

      const beepRate = Math.max(1, 16 - (R / 35));
      playSeekerBeep(beepRate);
    }}

    // Trail and exhaust
    curM.trail.push({{ x: curM.x, y: curM.y }});
    if (curM.trail.length > 30) curM.trail.shift();

    for (let i = 0; i < 4; i++) {{
      const spread = (Math.random() - 0.5) * 8;
      curM.smokeParticles.push({{
        x: curM.x - Math.cos(curM.heading) * 14 + spread,
        y: curM.y - Math.sin(curM.heading) * 14 + spread,
        vx: -Math.cos(curM.heading) * (70 + Math.random() * 50) + (Math.random() - 0.5) * 25,
        vy: -Math.sin(curM.heading) * (70 + Math.random() * 50) + (Math.random() - 0.5) * 25,
        radius: 3 + Math.random() * 4,
        alpha: 0.95,
        color: item.isFlanker ? 'rgba(192, 132, 252,' : (Math.random() > 0.3 ? 'rgba(255, 140, 0,' : 'rgba(255, 220, 150,')
      }});
    }}

    // Lethal proximity fuse radius: 38 px
    const isDirectHit = (missDist <= 38.0) || (R <= 38.0);
    if (isDirectHit) {{
      target.destroyed = true;
      missile.active = false;
      missileB.active = false;
      curM.stage = 'DESTROYED';
      playExplosionSound();

      // Massive kinetic explosion
      for (let i = 0; i < 95; i++) {{
        const angle = Math.random() * Math.PI * 2;
        const pSpeed = 90 + Math.random() * 340;
        explosionParticles.push({{
          x: target.x,
          y: target.y,
          vx: Math.cos(angle) * pSpeed,
          vy: Math.sin(angle) * pSpeed,
          radius: 2 + Math.random() * 5.5,
          life: 0.95 + Math.random() * 0.85,
          maxLife: 0.95 + Math.random() * 0.85,
          color: Math.random() > 0.5 ? '#F97316' : (Math.random() > 0.4 ? '#FBBF24' : '#EF4444')
        }});
      }}
      // Heavy mechanical debris
      for (let i = 0; i < 28; i++) {{
        debrisParticles.push({{
          x: target.x,
          y: target.y,
          vx: (Math.random() - 0.5) * 220,
          vy: (Math.random() - 0.5) * 220,
          rot: Math.random() * Math.PI * 2,
          rotSpeed: (Math.random() - 0.5) * 20,
          length: 6 + Math.random() * 14,
          life: 1.7
        }});
      }}

      document.getElementById('valStatus').innerText = 'SPLASH // TARGET DESTROYED (100% Pk)';
      document.getElementById('valStatus').style.color = '#34D399';
      document.getElementById('hudTTI').innerText = '0.0 s';
      document.getElementById('hudTTI').style.color = '#34D399';
      document.getElementById('hudSlantRange').innerText = '0 m (DIRECT HIT)';
      document.getElementById('hudBadge').innerText = 'THREAT NEUTRALIZED // AIRSPACE CLEAR';
      document.getElementById('hudBadge').style.color = '#34D399';
      document.getElementById('hudBadge').style.borderColor = 'rgba(52, 211, 153, 0.4)';
      document.getElementById('hudBadge').style.background = 'rgba(52, 211, 153, 0.18)';
      document.getElementById('threatPulse').style.background = '#34D399';
      document.getElementById('threatPulse').style.boxShadow = '0 0 12px #34D399';
      break;
    }}
  }}

  // Update particles for both missiles
  const allSmoke = [...missile.smokeParticles, ...missileB.smokeParticles];
  [missile.smokeParticles, missileB.smokeParticles].forEach(smokeArr => {{
    for (let i = smokeArr.length - 1; i >= 0; i--) {{
      const p = smokeArr[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.alpha -= dt * 2.2;
      p.radius += dt * 7;
      if (p.alpha <= 0) smokeArr.splice(i, 1);
    }}
  }});

  for (let i = explosionParticles.length - 1; i >= 0; i--) {{
    const p = explosionParticles[i];
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vx *= 0.93;
    p.vy *= 0.93;
    p.life -= dt;
    if (p.life <= 0) explosionParticles.splice(i, 1);
  }}

  for (let i = debrisParticles.length - 1; i >= 0; i--) {{
    const d = debrisParticles[i];
    d.x += d.vx * dt;
    d.y += d.vy * dt;
    d.vy += 140 * dt;
    d.rot += d.rotSpeed * dt;
    d.life -= dt;
    if (d.life <= 0) debrisParticles.splice(i, 1);
  }}
}}


// =============================================================================
// RENDERING & VISUAL RADAR
// =============================================================================
function draw() {{
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;

  // Polar Range Rings
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  const rings = [0.2, 0.4, 0.6, 0.8, 1.0];
  rings.forEach((ratio, idx) => {{
    const r = (h * 0.45) * ratio;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(148, 163, 184, 0.45)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${{(idx + 1) * 400}}m`, cx + 6, cy - r + 11);
  }});

  // Crosshairs
  ctx.beginPath();
  ctx.moveTo(cx, 15); ctx.lineTo(cx, h - 15);
  ctx.moveTo(15, cy); ctx.lineTo(w - 15, cy);
  ctx.stroke();

  // Radar Sweeping Beam
  ctx.save();
  ctx.translate(cx, cy);
  const sweepGrad = ctx.createConicGradient(radarSweepAngle, 0, 0);
  sweepGrad.addColorStop(0, 'rgba(56, 189, 248, 0.12)');
  sweepGrad.addColorStop(0.08, 'rgba(56, 189, 248, 0.0)');
  sweepGrad.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
  ctx.fillStyle = sweepGrad;
  ctx.beginPath();
  ctx.arc(0, 0, h * 0.48, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Launcher Station
  const lp = getLauncherPos();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.beginPath();
  ctx.arc(lp.x, lp.y, 14, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = '#94A3B8';
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillText('C2 INTERCEPTOR BATTERY', lp.x, lp.y + 24);

  // Target Drawing
  if (!target.destroyed) {{
    // Trail
    if (target.trail.length > 1) {{
      ctx.beginPath();
      ctx.moveTo(target.trail[0].x, target.trail[0].y);
      for (let i = 1; i < target.trail.length; i++) {{
        ctx.lineTo(target.trail[i].x, target.trail[i].y);
      }}
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }}

    // Target Box & Silhouette
    ctx.save();
    ctx.translate(target.x, target.y);

    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 1.8;
    ctx.strokeRect(-16, -16, 32, 32);

    ctx.rotate(target.heading);

    if (target.type === 'Warplane' || target.type === 'JetFighter') {{
      // Realistic Supersonic Military Jet Fighter Silhouette
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.moveTo(18, 0); // Needle nose
      ctx.lineTo(5, -6); // Cockpit
      ctx.lineTo(-6, -18); // Left delta wing tip
      ctx.lineTo(-8, -6);
      ctx.lineTo(-15, -12); // Left stabilizer
      ctx.lineTo(-12, 0); // Exhaust nozzle
      ctx.lineTo(-15, 12); // Right stabilizer
      ctx.lineTo(-8, 6);
      ctx.lineTo(-6, 18); // Right delta wing tip
      ctx.lineTo(5, 6);
      ctx.closePath();
      ctx.fill();

      // Twin Afterburner Flumes
      ctx.fillStyle = '#F97316';
      ctx.beginPath();
      ctx.arc(-13, 0, 3, 0, Math.PI * 2);
      ctx.fill();
    }} else if (target.type === 'Kamikaze') {{
      // Delta-wing Kamikaze Drone
      ctx.fillStyle = '#F59E0B';
      ctx.beginPath();
      ctx.moveTo(14, 0);
      ctx.lineTo(-10, -14);
      ctx.lineTo(-6, 0);
      ctx.lineTo(-10, 14);
      ctx.closePath();
      ctx.fill();
    }} else {{
      // Tactical Quadcopter Drone
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(-10, -10); ctx.lineTo(10, 10);
      ctx.moveTo(10, -10); ctx.lineTo(-10, 10);
      ctx.stroke();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
      [[-10,-10], [10,-10], [-10,10], [10,10]].forEach(([rx, ry]) => {{
        ctx.beginPath();
        ctx.arc(rx, ry, 4, 0, Math.PI * 2);
        ctx.fill();
      }});
      ctx.fillStyle = '#EF4444';
      ctx.beginPath();
      ctx.arc(0, 0, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }}
    ctx.restore();

    ctx.fillStyle = '#F87171';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`TGT: ${{target.type.toUpperCase()}}`, target.x + 22, target.y - 6);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${{Math.round(target.speed * 0.9)}} m/s | ALT ${{target.altitude}}m`, target.x + 22, target.y + 7);
  }}

  // Smoke particles from all active missiles
  [...missile.smokeParticles, ...missileB.smokeParticles].forEach(p => {{
    ctx.fillStyle = `${{p.color}}${{p.alpha}})`;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
    ctx.fill();
  }});

  // Render Interceptor Missiles (Alpha & Bravo Salvo)
  const drawList = [
    {{ m: missile, stroke: 'rgba(56, 189, 248, 0.85)', beam: 'rgba(56, 189, 248, 0.35)', name: 'ALPHA' }},
    {{ m: missileB, stroke: 'rgba(192, 132, 252, 0.85)', beam: 'rgba(192, 132, 252, 0.35)', name: 'BRAVO' }}
  ];

  drawList.forEach(item => {{
    const curM = item.m;
    if (curM.active && !target.destroyed) {{
      if (curM.trail.length > 1) {{
        ctx.beginPath();
        ctx.moveTo(curM.trail[0].x, curM.trail[0].y);
        for (let i = 1; i < curM.trail.length; i++) {{
          ctx.lineTo(curM.trail[i].x, curM.trail[i].y);
        }}
        ctx.strokeStyle = item.stroke;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }}

      // Target Intercept Line
      ctx.beginPath();
      ctx.moveTo(curM.x, curM.y);
      ctx.lineTo(target.x, target.y);
      ctx.strokeStyle = item.beam;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Missile Body
      ctx.save();
      ctx.translate(curM.x, curM.y);
      ctx.rotate(curM.heading);

      // Motor flare
      ctx.fillStyle = '#F97316';
      ctx.beginPath();
      ctx.moveTo(-10, -4);
      ctx.lineTo(-24 - Math.random() * 10, 0);
      ctx.lineTo(-10, 4);
      ctx.closePath();
      ctx.fill();

      // Rocket Body
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath();
      ctx.moveTo(14, 0); // cone
      ctx.lineTo(-8, -4);
      ctx.lineTo(-12, -8); // fin
      ctx.lineTo(-9, -3);
      ctx.lineTo(-9, 3);
      ctx.lineTo(-12, 8); // fin
      ctx.lineTo(-8, 4);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    }}
  }});

  // Explosion Particles
  explosionParticles.forEach(p => {{
    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.radius * (p.life / p.maxLife), 0, Math.PI * 2);
    ctx.fill();
  }});

  debrisParticles.forEach(d => {{
    ctx.save();
    ctx.translate(d.x, d.y);
    ctx.rotate(d.rot);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-d.length / 2, 0);
    ctx.lineTo(d.length / 2, 0);
    ctx.stroke();
    ctx.restore();
  }});

  // Splash Banner
  if (target.destroyed) {{
    ctx.save();
    ctx.fillStyle = 'rgba(16, 185, 129, 0.20)';
    ctx.fillRect(w * 0.12, h * 0.38, w * 0.76, 76);
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(w * 0.12, h * 0.38, w * 0.76, 76);

    ctx.fillStyle = '#34D399';
    ctx.font = '700 17px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('💥 SPLASH // TARGET NEUTRALIZED (DIRECT HIT)', w / 2, h * 0.38 + 32);

    ctx.fillStyle = '#E2E8F0';
    ctx.font = '11.5px "JetBrains Mono", monospace';
    ctx.fillText('KINETIC INTERCEPT CONFIRMED • MISS DISTANCE: 0.0m • EVASION THWARTED', w / 2, h * 0.38 + 56);
    ctx.restore();
  }}
}}

function loop(timestamp) {{
  const dt = Math.min(0.05, (timestamp - lastTimestamp) / 1000.0);
  lastTimestamp = timestamp;

  update(dt);
  draw();

  requestAnimationFrame(loop);
}}

initScenario();
requestAnimationFrame(loop);

document.getElementById('btnLaunch').addEventListener('click', launchInterceptor);
document.getElementById('btnReset').addEventListener('click', initScenario);
document.getElementById('selectTargetType').addEventListener('change', initScenario);
document.getElementById('selectEvasion').addEventListener('change', (e) => {{
  target.evasionMode = e.target.value;
}});

const audioBtn = document.getElementById('btnAudioToggle');
audioBtn.addEventListener('click', () => {{
  soundEnabled = !soundEnabled;
  audioBtn.innerText = soundEnabled ? '🔊 AUDIO: ON' : '🔈 AUDIO: MUTED';
}});

</script>
</body>
</html>
"""
