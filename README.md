# Chinook Customer Support Agent

## Agent Architecture Diagram

```mermaid
flowchart TD
	U[User] --> APP[Gradio App]
	APP --> S[supervisor]

	S -->|primary route| IA[invoice_agent]
	S -->|primary route| CA[catalog_agent]
	S -->|primary route| MA[memory_agent]

	S -. if primary != memory_agent,<br/>run in parallel .-> MA

	IA <--> IT[invoice_tools]
	CA <--> CT[catalog_tools]
	MA <--> MT[memory_tools]

	IA --> END((END))
	CA --> END
	MA --> END

	CT -. handoff .-> IA
	IT -. handoff .-> CA
	MT -. handoff .-> IA
	MT -. handoff .-> CA
```

Image version: [docs/agent-architecture-diagram.svg](docs/agent-architecture-diagram.svg)

## Request Execution Diagram

```mermaid
sequenceDiagram
	participant User
	participant App as Gradio App
	participant Sup as supervisor
	participant Prim as Primary Agent
	participant Mem as memory_agent
	participant Tools as ToolNode

	User->>App: Send message
	App->>Sup: invoke(state)
	Sup->>Sup: Structured routing decision

	alt next_agent == memory_agent
		Sup->>Mem: Run as primary
		loop Until no tool calls
			Mem->>Tools: optional tool call
			Tools-->>Mem: tool result
		end
		Mem-->>App: response
	else next_agent == invoice_agent or catalog_agent
		par Primary execution
			Sup->>Prim: Run selected primary agent
			loop Until no tool calls
				Prim->>Tools: optional tool call
				Tools-->>Prim: tool result
			end
			Prim-->>App: primary response
		and Background sync
			Sup->>Mem: Run in parallel
			loop optional memory tool calls
				Mem->>Tools: save/get preferences
				Tools-->>Mem: tool result
			end
		end
	end

	App-->>User: Final response
```

Image version: [docs/request-execution-flow.svg](docs/request-execution-flow.svg)

## Quick Start

1. Clone and enter the repository.

```bash
git clone https://github.com/kansaram/chinook-customer-support-agent.git
cd chinook-customer-support-agent
```

2. Create a Python 3.12 virtual environment.

```bash
py -3.12 -m venv .venv
```

3. Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies.

```bash
pip install -r requirements.txt
```

5. Configure environment variables.

```bash
cp .env.example .env
```

Add your `OPENAI_API_KEY` to `.env`.

## Run The App

From the project root:

```bash
python src/chinook_agent/app.py
```

Default URL: `http://127.0.0.1:7860`

Optional server overrides:

- `GRADIO_SERVER_NAME`
- `GRADIO_SERVER_PORT`

Example:

```powershell
$env:GRADIO_SERVER_NAME="0.0.0.0"
$env:GRADIO_SERVER_PORT="7860"
python src/chinook_agent/app.py
```

## Verify Local Databases

List tables:

```bash
python -c "import sqlite3; c = sqlite3.connect('data/chinook.db'); print(c.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall())"
```

Check customer count:

```bash
python -c "import sqlite3; c = sqlite3.connect('data/chinook.db'); print(c.execute('SELECT COUNT(*) FROM customers').fetchone())"
```

## Run Tests

```bash
pytest tests/ -v
```

## Docker

Build image:

```bash
docker build -t chinook-customer-support-agent .
```

Run container:

```bash
docker run -d -p 7860:7860 --name chinook-customer-support-agent --env-file .env chinook-customer-support-agent
```

Open: `http://localhost:7860`


