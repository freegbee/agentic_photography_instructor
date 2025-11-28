# MLflow SFTP Artifact Storage Setup

This document describes the setup for using SFTP to store MLflow artifacts on a remote server.

## Configuration

### SFTP Server Details
- **Host**: tajo.host.ch
- **Port**: 2121
- **User**: freegee
- **Authentication**: SSH key (stored in `docker/.secrets/mlflow_sftp_key/id_rsa`)

### Files Configured

1. **SSH Configuration** (`docker/mlflow/ssh_config`)
   - Defines connection parameters for tajo.host.ch
   - Specifies the SSH key location

2. **SSH Key Initialization** (`docker/mlflow/init_ssh_keys.sh`)
   - Adds the SFTP host to known_hosts
   - Runs automatically on container startup

3. **Environment Variables** (`docker/.env`)
   - `MLFLOW_ARTIFACT_ROOT`: SFTP URI for artifact storage
   - `MLFLOW_ARTIFACT_URI`: Same as MLFLOW_ARTIFACT_ROOT (for compatibility)

## Activation Steps

### 1. Configure Environment Variables

Edit your `docker/.env` file (copy from `docker/.env.dist` if needed):

```bash
# Set these variables:
MLFLOW_ARTIFACT_ROOT=sftp://freegee@tajo.host.ch:2121/agent.freegee.ch/artifacts
MLFLOW_ARTIFACT_URI=sftp://freegee@tajo.host.ch:2121/agent.freegee.ch/artifacts
```

**Note**: You can customize the remote path (`/mlflow/artifacts`) to any directory on your SFTP server where you have write permissions.

### 2. Ensure SSH Key is in Place

Make sure your SSH private key is at:
```
docker/.secrets/mlflow_sftp_key/id_rsa
```

The key should have proper permissions (this is set automatically in containers):
- Private key: 600 (read/write for owner only)
- `.ssh` directory: 700

### 3. Rebuild and Restart Containers

Since we've updated Dockerfiles to include SSH client dependencies, rebuild the containers:

```bash
cd docker

# Rebuild all containers
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

Or rebuild specific services:
```bash
docker-compose build mlflow notebook experiment-service train
docker-compose up -d mlflow notebook experiment-service train
```

### 4. Verify SFTP Connection

Test the SFTP connection from within a container:

```bash
# Test from mlflow container
docker exec -it mlflow bash
sftp -P 2121 freegee@tajo.host.ch

# Test from experiment-service container
docker exec -it experiment-service bash
sftp -P 2121 freegee@tajo.host.ch
```

If successful, you should get an SFTP prompt without password authentication.

### 5. Test MLflow Artifact Storage

Run a simple experiment to verify artifacts are stored on SFTP:

```bash
# Attach to experiment-service
docker exec -it experiment-service bash

# Run an experiment
python -m experiments.image_scoring.entrypoint
```

Check your SFTP server at the configured path to verify artifacts were uploaded.

## How It Works

### Container Initialization

Each container that needs SFTP access (mlflow, notebook, experiment-service, train) runs this initialization on startup:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
chmod 600 /root/.ssh/id_rsa
/init_ssh_keys.sh  # Adds host to known_hosts
```

### Mounted Files

All containers mount:
- `./secrets/mlflow_sftp_key/id_rsa` → `/root/.ssh/id_rsa` (read-only)
- `./mlflow/ssh_config` → `/root/.ssh/config` (read-only)
- `./mlflow/init_ssh_keys.sh` → `/init_ssh_keys.sh` (read-only)

### Environment Variables

All containers receive:
- `MLFLOW_ARTIFACT_ROOT`: The SFTP URI for artifact storage
- `MLFLOW_ARTIFACT_URI`: Same URI for compatibility with different MLflow client versions

## Troubleshooting

### Connection Refused

If you get "Connection refused":
1. Verify the SFTP server is running on port 2121
2. Check firewall rules allow connections to port 2121
3. Verify the host is reachable: `ping tajo.host.ch`

### Permission Denied

If you get "Permission denied (publickey)":
1. Verify the SSH key in `docker/.secrets/mlflow_sftp_key/id_rsa` is correct
2. Check that the public key is authorized on the SFTP server
3. Verify the key format is correct (OpenSSH format)

### Path Not Found

If MLflow fails to write artifacts:
1. Verify the path `/mlflow/artifacts` exists on the SFTP server
2. Or change the path in `MLFLOW_ARTIFACT_ROOT` to a valid directory
3. Ensure the user `freegee` has write permissions to that directory

### Host Key Verification Failed

If you get "Host key verification failed":
1. The `init_ssh_keys.sh` script should prevent this
2. Check the script is executable and runs correctly
3. Manually add the host key: `ssh-keyscan -p 2121 tajo.host.ch >> ~/.ssh/known_hosts`

## Switching Between Local and SFTP Storage

To switch back to local artifact storage, modify `docker/.env`:

```bash
# Local storage (in Docker volume)
MLFLOW_ARTIFACT_ROOT=/mlruns
MLFLOW_ARTIFACT_URI=/mlruns
```

Then restart containers:
```bash
cd docker
docker-compose restart mlflow notebook experiment-service train
```

## Security Notes

1. **SSH Key Protection**:
   - The `.secrets/` directory should be in `.gitignore`
   - Never commit SSH private keys to version control
   - Keep proper file permissions (600 for private key)

2. **Read-Only Mounts**:
   - SSH credentials are mounted read-only (`:ro`) in containers
   - This prevents accidental modification

3. **Host Key Verification**:
   - The `StrictHostKeyChecking accept-new` setting in `ssh_config` accepts new host keys automatically
   - For maximum security, you could set this to `yes` after first connection

## Additional Information

- MLflow uses the `paramiko` library for SFTP connections
- The SFTP URI format is: `sftp://user@host:port/path/to/artifacts`
- Artifacts are stored directly on the SFTP server, bypassing local storage
- The MLflow tracking server (`--serve-artifacts`) proxies artifact requests to SFTP
