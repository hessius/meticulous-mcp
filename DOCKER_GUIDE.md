# Meticulous MCP Server: Docker Guide

This guide explains how to build and run the Meticulous MCP server using Docker.

## Prerequisites

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/) installed.
- A Meticulous Home Espresso machine on your local network.

## Quick Start (Build & Run)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manonstreet/meticulous-mcp.git
   cd meticulous-mcp
   ```

2. **Configure your machine IP:**
   Open `docker-compose.yml` and replace the `METICULOUS_API_URL` with your machine's local IP address:
   ```yaml
   environment:
     # Examples: http://192.168.x.x, http://10.x.x.x, http://172.16.x.x
     - METICULOUS_API_URL=http://<YOUR_MACHINE_IP>
   ```

3. **Build and start the container:**
   ```bash
   docker compose up -d --build --wait
   ```
   The `--wait` flag ensures the command only finishes once the server is healthy and ready to accept connections. The server will run in the background on port `8080`.

## Verifying the Server

If you didn't use the `--wait` flag, or want to see live activity, you can check the logs:
```bash
docker compose logs -f
```

## Integrating with Claude Desktop

You can connect Claude Desktop to this server in two ways.

### Option 1: SSE / HTTP (Recommended)
This method works whether the Docker container is running locally or on a remote machine on your network.

Replace `<HOST_IP>` with the IP address or hostname of the machine running the container (use `localhost` if running on the same machine):

```json
{
  "mcpServers": {
    "meticulous": {
      "url": "http://<HOST_IP>:8080/mcp"
    }
  }
}
```

### Option 2: Docker Exec (Local Only)
If the Docker container is running on the **same machine** as Claude Desktop, you can use the direct `docker exec` method. This communicates via standard input/output (stdio).

```json
{
  "mcpServers": {
    "meticulous": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "meticulous-mcp-server",
        "python",
        "meticulous-mcp/run_server.py"
      ]
    }
  }
}
```
*Note: This assumes the container name is `meticulous-mcp-server` (default in docker-compose.yml).*

## Troubleshooting

- **Connection Refused:** Ensure your Meticulous machine is awake and on the same Wi-Fi network.
- **Port Conflict:** If port `8080` is already in use, change the mapping in `docker-compose.yml`.
