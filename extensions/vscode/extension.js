let vscode;
try {
  vscode = require('vscode');
} catch (_) {
  vscode = undefined;
}
const cp = require('child_process');

let currentProcess;
let provider;
const state = {
  mode: 'idle',
  status: 'Ready. Run a preview command to inspect generated output before mutating Git.',
  report: null,
  message: '',
  output: '',
  error: '',
  prOutput: '',
  prError: '',
};

function activate(context) {
  ensureVscode();
  provider = new AutocommitReviewProvider(context.extensionUri);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('autocommit.review', provider),
    vscode.commands.registerCommand('autocommit.analyze', analyze),
    vscode.commands.registerCommand('autocommit.generateCommitMessage', generateCommitMessage),
    vscode.commands.registerCommand('autocommit.commitApproved', commitApproved),
    vscode.commands.registerCommand('autocommit.prPreview', prPreview),
    vscode.commands.registerCommand('autocommit.prCreate', prCreate),
  );
}

function deactivate() {
  cancelCurrentProcess();
}

async function analyze() {
  await runAutocommitPreview('analyze', ['analyze', '--json'], (stdout) => {
    const report = parseJson(stdout);
    state.report = report;
    state.message = composeCommitMessage(report);
    state.output = formatAnalysis(report);
    state.status = 'Preview complete. No Git changes were made.';
  });
}

async function generateCommitMessage() {
  const args = ['commit', '--dry-run', '--json', '--no-interactive'];
  if (config().get('commitStagedOnly')) args.push('--staged');
  await runAutocommitPreview('commit preview', args, (stdout) => {
    const report = parseJson(stdout);
    state.report = report;
    state.message = composeCommitMessage(report);
    state.output = formatAnalysis(report);
    state.status = 'Commit message generated for review. Edit it before committing if needed.';
  });
}

async function prPreview() {
  await runAutocommitPreview('PR dry run', ['pr', '--dry-run', '--no-interactive'], (stdout) => {
    state.prOutput = stdout.trim();
    state.prError = '';
    state.status = 'PR dry run complete. No pull request was created.';
  });
}

async function prCreate() {
  const workspace = workspaceFolder();
  if (!workspace) return;

  const base = config().get('prBase') || '';
  const draft = config().get('prDraft');
  const reviewers = config().get('prReviewers') || [];
  const labels = config().get('prLabels') || [];
  const extraArgs = config().get('extraArgs') || [];

  const args = ['pr', '--interactive'];
  if (draft) args.push('--draft');
  if (base) args.push('--base', base);
  for (const r of reviewers) args.push('--reviewer', r);
  for (const l of labels) args.push('--label', l);
  args.push(...extraArgs);

  const terminal = vscode.window.createTerminal({
    name: 'autocommit PR',
    message: 'Creating PR with autocommit...',
  });
  terminal.show();
  terminal.sendText(`${shellQuote(config().get('binaryPath'))} ${args.join(' ')}`);
}

async function prUpdate() {
  ensureVscode();
  vscode.window.showInformationMessage(
    'Run `autocommit pr` in the terminal to update an existing PR with new issue links and metadata.',
  );
}

async function commitApproved(messageFromView) {
  const message = (messageFromView || state.message || '').trim();
  if (!message) {
    vscode.window.showErrorMessage(
      'No approved autocommit message is available. Generate a commit message preview first.',
    );
    return;
  }

  const workspace = workspaceFolder();
  if (!workspace) return;

  const stagedOnly = config().get('commitStagedOnly');
  if (!stagedOnly) {
    const stage = await vscode.window.showWarningMessage(
      'Commit Approved Message mutates Git. Stage all current changes before committing?',
      { modal: true },
      'Stage All and Commit',
      'Commit Staged Only',
    );
    if (!stage) return;
    if (stage === 'Stage All and Commit') await runGit(['add', '-A'], workspace.uri.fsPath);
  }

  const hasStagedChanges = await gitHasStagedChanges(workspace.uri.fsPath);
  if (!hasStagedChanges) {
    vscode.window.showErrorMessage('No staged changes are available to commit.');
    return;
  }

  const binaryPath = config().get('binaryPath');
  const cliArgs = ['commit', '-m', message, '--no-interactive', '--yes', '--staged'];

  update({ mode: 'running', status: 'Creating commit with the approved message...', error: '' });
  try {
    const stdout = await runProcess(binaryPath, cliArgs, workspace.uri.fsPath);
    update({
      mode: 'idle',
      status: 'Commit created from approved message.',
      output: stdout.trim(),
    });
    vscode.window.showInformationMessage('autocommit: commit created from approved message.');
  } catch (err) {
    update({
      mode: 'error',
      status: 'Commit failed.',
      error: String(err.message || err),
    });
    vscode.window.showErrorMessage(`autocommit commit failed: ${err.message || err}`);
  }
}

async function runAutocommitPreview(label, baseArgs, onSuccess) {
  const workspace = workspaceFolder();
  if (!workspace) return;
  cancelCurrentProcess();
  update({
    mode: 'running',
    status: `Running autocommit ${label}...`,
    output: '',
    error: '',
  });
  const args = baseArgs.concat(config().get('extraArgs') || []);
  try {
    const stdout = await runProcess(config().get('binaryPath'), args, workspace.uri.fsPath);
    onSuccess(stdout);
    update({ mode: 'idle' });
  } catch (err) {
    const isBinaryMissing =
      String(err.message || err).includes('ENOENT') ||
      String(err.message || err).includes('not found');
    const hint = isBinaryMissing
      ? ' Is autocommit installed? Set autocommit.binaryPath in settings.'
      : '';
    update({
      mode: 'error',
      status: `autocommit ${label} failed.${hint}`,
      error: String(err.message || err),
    });
    vscode.window.showErrorMessage(`autocommit ${label} failed:${hint}`);
  }
}

function workspaceFolder() {
  ensureVscode();
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  if (!folder) vscode.window.showErrorMessage('Open a workspace folder before running autocommit.');
  return folder;
}

function config() {
  ensureVscode();
  return vscode.workspace.getConfiguration('autocommit');
}

function ensureVscode() {
  if (!vscode) throw new Error('The VS Code API is only available inside the extension host.');
}

function runProcess(command, args, cwd) {
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    currentProcess = cp.spawn(command, args, { cwd, shell: false });
    currentProcess.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
      update({ output: stdout });
    });
    currentProcess.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
      update({ error: stderr });
    });
    currentProcess.on('error', reject);
    currentProcess.on('close', (code, signal) => {
      currentProcess = undefined;
      if (signal) reject(new Error(`process canceled with signal ${signal}`));
      else if (code === 0) resolve(stdout);
      else {
        const msg = stderr.trim() || stdout.trim() || `process exited with code ${code}`;
        reject(new Error(msg));
      }
    });
  });
}

function runGit(args, cwd) {
  return runProcess('git', args, cwd);
}

async function gitHasStagedChanges(cwd) {
  try {
    await runGit(['diff', '--cached', '--quiet'], cwd);
    return false;
  } catch (_) {
    return true;
  }
}

function cancelCurrentProcess() {
  if (currentProcess) currentProcess.kill();
}

function parseJson(stdout) {
  try {
    return JSON.parse(stdout);
  } catch (err) {
    throw new Error(`expected JSON from autocommit: ${err.message}`);
  }
}

function composeCommitMessage(report) {
  if (!report) return '';
  const body = report.body ? `\n\n${report.body}` : '';
  const lines = [report.commit_message || 'chore: update changes', ''];
  if (report.summary) lines.push(report.summary, '');
  if (Array.isArray(report.items) && report.items.length) {
    lines.push('### Changes');
    for (const item of report.items) {
      const path =
        item.files && item.files[0] && item.files[0].path ? ` [${item.files[0].path}]` : '';
      lines.push(`- ${item.title || item.intent || 'Update'}${path}`);
    }
    lines.push('');
  }
  if (report.risk && Array.isArray(report.risk.notes) && report.risk.notes.length) {
    lines.push(`### Risk (${report.risk.level || 'unknown'})`);
    for (const note of report.risk.notes) lines.push(`- ${note}`);
  }
  let result = lines.join('\n').trim();
  if (body) result += body;
  return result;
}

function formatAnalysis(report) {
  if (!report) return '';
  const stats = report.stats || {};
  const risk = report.risk || {};
  return [
    `Summary: ${report.summary || 'n/a'}`,
    `Risk: ${risk.level || 'unknown'}`,
    `Dispatch: ${report.dispatch && report.dispatch.route ? report.dispatch.route : 'unknown'}`,
    `Files changed: ${stats.files_changed || 0}`,
    '',
    'Items:',
    ...(report.items || []).map((item) => `- ${item.title || item.intent || item.id}`),
  ].join('\n');
}

function buildGitCommitArgs(message) {
  const lines = message.split(/\r?\n/);
  const subject = lines.shift().trim();
  const body = lines.join('\n').trim();
  const args = ['commit', '-m', subject];
  if (body) args.push('-m', body);
  return args;
}

function shellQuote(value) {
  return String(value || '')
    .replace(/'/g, `'\\''`)
    .replace(/^(.+)$/, "'$1'");
}

function update(patch) {
  Object.assign(state, patch);
  if (provider) provider.refresh();
}

class AutocommitReviewProvider {
  constructor(extensionUri) {
    this.extensionUri = extensionUri;
  }

  resolveWebviewView(webviewView) {
    this.view = webviewView;
    webviewView.webview.options = { enableScripts: true };
    webviewView.webview.onDidReceiveMessage((message) => {
      if (message.command === 'analyze') analyze();
      if (message.command === 'generate') generateCommitMessage();
      if (message.command === 'commit') commitApproved(message.text);
      if (message.command === 'cancel') cancelCurrentProcess();
      if (message.command === 'updateMessage') state.message = message.text || '';
      if (message.command === 'prPreview') prPreview();
      if (message.command === 'prCreate') prCreate();
      if (message.command === 'prUpdate') prUpdate();
    });
    this.refresh();
  }

  refresh() {
    if (this.view) this.view.webview.html = renderHtml(this.view.webview);
  }
}

function renderHtml(webview) {
  const nonce = String(Date.now());
  const busy = state.mode === 'running';
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>
body { padding: 12px; color: var(--vscode-foreground); font-family: var(--vscode-font-family); }
button { margin: 0 6px 8px 0; }
textarea { width: 100%; min-height: 180px; color: var(--vscode-input-foreground); background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border); }
pre { white-space: pre-wrap; border: 1px solid var(--vscode-panel-border); padding: 8px; max-height: 260px; overflow: auto; }
.status { margin: 8px 0; }
.danger { color: var(--vscode-errorForeground); font-weight: 600; }
.small { opacity: 0.8; font-size: 0.9em; }
.section { margin-top: 16px; border-top: 1px solid var(--vscode-panel-border); padding-top: 8px; }
</style>
</head>
<body>
<h2>autocommit review</h2>
<div class="status">${escapeHtml(state.status)}</div>
<button ${busy ? 'disabled' : ''} id="analyze">Analyze (Preview)</button>
<button ${busy ? 'disabled' : ''} id="generate">Generate Commit Message (Preview)</button>
<button ${busy ? '' : 'disabled'} id="cancel">Cancel</button>
<p class="small">Preview actions run analysis only; they do not mutate Git.</p>
<label for="message">Approved commit message</label>
<textarea id="message">${escapeHtml(state.message)}</textarea>
<p><button ${busy || !state.message ? 'disabled' : ''} id="commit">Commit Approved Message</button> <span class="danger">Mutation: creates a Git commit.</span></p>
<h3>Generated output</h3>
<pre>${escapeHtml(state.output || 'No output yet.')}</pre>
${state.error ? `<h3 class="danger">Errors</h3><pre>${escapeHtml(state.error)}</pre>` : ''}
<div class="section">
<h3>Pull Request</h3>
<button ${busy ? 'disabled' : ''} id="prPreview">PR Preview (Dry Run)</button>
<button ${busy ? 'disabled' : ''} id="prCreate">Create PR...</button>
<button ${busy ? 'disabled' : ''} id="prUpdate">Update PR</button>
<p class="small">PR commands open a terminal for interactive input (issue linking, reviewers, etc.).</p>
${state.prOutput ? `<pre>${escapeHtml(state.prOutput)}</pre>` : ''}
${state.prError ? `<h3 class="danger">PR Errors</h3><pre>${escapeHtml(state.prError)}</pre>` : ''}
</div>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
document.getElementById('analyze').addEventListener('click', () => vscode.postMessage({ command: 'analyze' }));
document.getElementById('generate').addEventListener('click', () => vscode.postMessage({ command: 'generate' }));
document.getElementById('cancel').addEventListener('click', () => vscode.postMessage({ command: 'cancel' }));
document.getElementById('commit').addEventListener('click', () => vscode.postMessage({ command: 'commit', text: document.getElementById('message').value }));
document.getElementById('message').addEventListener('input', (event) => vscode.postMessage({ command: 'updateMessage', text: event.target.value }));
document.getElementById('prPreview').addEventListener('click', () => vscode.postMessage({ command: 'prPreview' }));
document.getElementById('prCreate').addEventListener('click', () => vscode.postMessage({ command: 'prCreate' }));
document.getElementById('prUpdate').addEventListener('click', () => vscode.postMessage({ command: 'prUpdate' }));
</script>
</body>
</html>`;
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

module.exports = {
  activate,
  deactivate,
  composeCommitMessage,
  formatAnalysis,
  buildGitCommitArgs,
};
