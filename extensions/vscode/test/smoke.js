const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
const extension = require(path.join(root, 'extension.js'));

const commands = new Set(pkg.contributes.commands.map((command) => command.command));
for (const expected of [
  'autocommit.analyze',
  'autocommit.generateCommitMessage',
  'autocommit.commitApproved',
  'autocommit.prPreview',
  'autocommit.prCreate',
]) {
  assert(commands.has(expected), `missing command contribution: ${expected}`);
}

const report = {
  commit_message: 'feat(vscode): add review panel',
  summary: 'Adds a VS Code review flow.',
  body: 'Implements a full review loop with preview-then-approve.',
  items: [
    { title: 'Review generated output', files: [{ path: 'extensions/vscode/extension.js' }] },
  ],
  risk: { level: 'low', notes: ['Manual QA required in VS Code.'] },
  stats: { files_changed: 2 },
  dispatch: { route: 'Full' },
};

const message = extension.composeCommitMessage(report);
assert(message.startsWith('feat(vscode): add review panel'));
assert(message.includes('### Changes'));
assert(message.includes('extensions/vscode/extension.js'));
assert(message.includes('### Risk (low)'));
assert(message.includes('Implements a full review loop'), 'body field should be included');

assert.deepStrictEqual(extension.buildGitCommitArgs('subject\n\nbody line'), [
  'commit',
  '-m',
  'subject',
  '-m',
  'body line',
]);
assert(extension.formatAnalysis(report).includes('Files changed: 2'));

console.log('VS Code extension smoke checks passed');
