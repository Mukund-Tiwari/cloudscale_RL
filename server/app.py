from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import the updated environment
from server.cloudscale_RL_environment import CloudAutoScalerEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CloudScale Admin — OpenEnv API",
    version="1.0.0",
    description="Scalable RL Environment for microservice autoscaling and traffic spike management.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Web Dashboard HTML
# ---------------------------------------------------------------------------
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloudScale RL Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 { 
            color: #333; 
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle { 
            color: #666; 
            margin-bottom: 30px; 
            font-size: 1.1em;
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px;
            margin-bottom: 30px;
        }
        .card { 
            background: #f8f9fa; 
            border: 1px solid #e0e0e0;
            border-radius: 8px; 
            padding: 20px; 
        }
        .card h2 { 
            color: #667eea; 
            margin-bottom: 15px; 
            font-size: 1.3em;
        }
        .endpoint { 
            background: white;
            border-left: 4px solid #667eea;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
        }
        .method { 
            display: inline-block; 
            padding: 4px 8px; 
            border-radius: 3px; 
            margin-right: 10px;
            font-weight: bold;
            font-size: 0.85em;
        }
        .post { background: #667eea; color: white; }
        .get { background: #48bb78; color: white; }
        button { 
            background: #667eea; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 10px;
            transition: background 0.3s;
        }
        button:hover { background: #764ba2; }
        .response { 
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-top: 10px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .status { padding: 10px; border-radius: 4px; margin-top: 10px; }
        .success { background: #c6f6d5; color: #22543d; border: 1px solid #9ae6b4; }
        .error { background: #fed7d7; color: #742a2a; border: 1px solid #fc8181; }
        .links { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
        a { 
            color: #667eea; 
            text-decoration: none; 
            padding: 10px 15px;
            border: 1px solid #667eea;
            border-radius: 4px;
            transition: all 0.3s;
            display: inline-block;
        }
        a:hover { background: #667eea; color: white; }
        ul { list-style: none; }
        ul li { margin: 8px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 CloudScale RL Environment</h1>
        <p class="subtitle">Multi-Agent Cloud Autoscaling RL Environment</p>
        
        <div class="grid">
            <div class="card">
                <h2>📊 API Endpoints</h2>
                <div class="endpoint">
                    <span class="method post">POST</span>/reset
                    <br><small>Start a new episode</small>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>/step
                    <br><small>Take an action and advance environment</small>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>/health
                    <br><small>Check server health</small>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>/state
                    <br><small>Get current global state</small>
                </div>
            </div>

            <div class="card">
                <h2>⚙️ Quick Test</h2>
                <p>Test the /reset endpoint directly:</p>
                <button onclick="testReset()">Test /reset Endpoint</button>
                <div id="resetResponse" class="response" style="display:none;"></div>
            </div>

            <div class="card">
                <h2>📚 Documentation & Tools</h2>
                <div class="links">
                    <a href="/docs">📖 Interactive API Docs</a>
                    <a href="/redoc">📕 Alternative Docs</a>
                </div>
            </div>

            <div class="card">
                <h2>ℹ️ Environment Info</h2>
                <ul>
                    <li><strong>Tasks:</strong> easy, medium, hard</li>
                    <li><strong>Services:</strong> frontend, backend, worker</li>
                    <li><strong>Max Steps:</strong> 200</li>
                    <li><strong>Actions:</strong> SCALE_UP, SCALE_DOWN, NO_OP</li>
                </ul>
                <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    Manage cloud resources automatically during traffic spikes and maintain SLA compliance.
                </p>
            </div>
        </div>

        <div class="card">
            <h2>🔗 External Links</h2>
            <ul>
                <li><a href="https://github.com/piyushagarwal0317-crypto/cloudscale_RL" target="_blank">📦 GitHub Repository</a></li>
                <li><a href="https://huggingface.co/spaces/bitmain/cloudscale_RL" target="_blank">🤗 HF Space Page</a></li>
            </ul>
        </div>
    </div>

    <script>
        async function testReset() {
            const responseDiv = document.getElementById('resetResponse');
            try {
                const response = await fetch('/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: '{}'
                });
                const data = await response.json();
                responseDiv.textContent = JSON.stringify(data, null, 2);
                responseDiv.className = 'response success';
                responseDiv.style.display = 'block';
            } catch (error) {
                responseDiv.textContent = 'Error: ' + error.message;
                responseDiv.className = 'response error';
                responseDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
_env: Optional[CloudAutoScalerEnv] = None
_last_obs = None

def get_env(task_level: str = "medium") -> CloudAutoScalerEnv:
    global _env
    # Re-initialize if the environment doesn't exist OR if the task level changed
    if _env is None or _env.task_level != task_level:
        _env = CloudAutoScalerEnv(
            task_level=task_level,
            max_steps=MAX_STEPS
        )
    return _env

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------
class StepRequest(BaseModel):
    actions: Dict[str, str] = Field(
        default={"frontend": "NO_OP", "backend": "NO_OP", "worker": "NO_OP"},
        description="Scaling actions for each microservice agent."
    )

class ResetRequest(BaseModel):
    task_level: str = Field(
        default="medium",
        description="Difficulty of the task: 'easy', 'medium', or 'hard'."
    )

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _obs_to_dict(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        agent_id: {
            "active_pods": obs.get("active_pods"),
            "pending_pods": obs.get("pending_pods"),
            "rps": obs.get("rps"),
            "latency_p95": obs.get("latency_p95"),
            "error_rate": obs.get("error_rate"),
            "queue_depth": obs.get("queue_depth"),
            "utilization": obs.get("utilization"),
            "spike_detected": obs.get("spike_detected")
        }
        for agent_id, obs in obs_dict.items()
    }

def _reward_to_dict(rewards_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        agent_id: {
            "total": reward.get("total", 0.0),
            "components": {
                "r_latency": reward.get("r_latency", 0.0),
                "r_cost": reward.get("r_cost", 0.0),
                "r_action": reward.get("r_action", 0.0),
                "r_spike": reward.get("r_spike", 0.0),
                "r_sla": reward.get("r_sla", 0.0)
            }
        }
        for agent_id, reward in rewards_dict.items()
    }

# ---------------------------------------------------------------------------
# Web Dashboard Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web dashboard at the root URL."""
    return HTML_DASHBOARD

@app.get("/web", response_class=HTMLResponse)
async def web():
    """Alternative web endpoint for HF Space compatibility."""
    return HTML_DASHBOARD

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    env = get_env()
    return {
        "status": "ok",
        "system_healthy": env.stats.system_healthy,
        "uptime": time.time() - env.stats.start_time,
        "timestamp": time.time(),
    }

@app.post("/reset")
def reset(req: ResetRequest = None):
    global _last_obs
    # Default to medium if no request body is provided
    task_level = req.task_level if req else "medium"
    
    env = get_env(task_level=task_level)
    obs = env.reset()
    _last_obs = obs
    return {
        "status": "reset", 
        "task_level": task_level,
        "observation": _obs_to_dict(obs)
    }

@app.post("/step")
def step(req: StepRequest):
    global _last_obs
    env = get_env()
    
    if _last_obs is None:
        env.reset()
        
    # FIX: Correctly unpack the CloudStepResult object
    result = env.step(req.actions)
    _last_obs = result.observations
    
    info = result.info
    
    # 🏆 KILLER FEATURE: Inject the Grader Score when the episode finishes
    if result.done:
        final_score = env.grade_task()
        final_score = max(0.01, min(0.99, final_score))
        info["final_score"] = final_score
        info["task_level"] = env.task_level
        logger.info(f"Episode finished. Task: {env.task_level} | Final Score: {info['final_score']}")
    
    return {
        "done": result.done,
        "info": info,
        "observation": _obs_to_dict(result.observations),
        "rewards": _reward_to_dict(result.rewards),
    }

@app.get("/state")
def get_state():
    env = get_env()
    obs = env.get_global_state()
    return _obs_to_dict(obs)

def main():
    """Main entry point for the server."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()