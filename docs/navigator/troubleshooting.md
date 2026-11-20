# Navigator troubleshooting

| Message | Cause and fix |
|---|---|
| `SSH needs the paramiko package` | Install `paramiko` for the Python that runs the server, then restart the server. |
| A finished job still shows *running*, and the chain does not continue | The server cannot ask the cluster: a key passphrase is missing, or a login failed. Enter the passphrase in **Job settings** (see [Enter the key passphrase](index.md#enter-the-key-passphrase)) or press **Retry** under [SSH connections](running-jobs.md#ssh-connections), then press **Refresh**. |
| `SSH to … is paused after a failed login` | The last login to that node failed and the server will not try again on its own. Fix the cause shown, then press **Retry** under [SSH connections](running-jobs.md#ssh-connections). |
| **Delete** does nothing / "not found" | The server is older than the page. Restart the server. |
| `wrong passphrase for ~/.ssh/id_navigator`, or `… is encrypted: enter its passphrase` | Wrong or missing **Key passphrase**. Use the key's passphrase, not your account password. Check it with `ssh-keygen -y -f ~/.ssh/id_navigator`. |
| `Authentication failed` | The public key is not in `~/.ssh/authorized_keys`, or the **User** is wrong. |
| `Connection reset by peer` / `Error reading SSH protocol banner` | The login node is refusing connections for a while, usually after many failed logins. Wait about 30 minutes. |
| `invalid partition specified` | The job was submitted on the wrong cluster. Check **Submit from** and **Login node**. |
| Page asks for a token | Paste the token the server printed at start-up. |
| `Address already in use` at start-up | Another program, for example another Navigator server, already uses port 8765 on this login node. Start with another port, such as `--port 8800`. |
