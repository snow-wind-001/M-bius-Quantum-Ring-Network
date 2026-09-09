#!/usr/bin/env node
// Use the installed CodeRecoder backup engine; Git and restore are separate.
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    'coderecoder-root': { type: 'string', default: path.resolve(project, '..', 'CodeRecoder') },
    'storage-root': { type: 'string', default: path.join(os.homedir(), 'CodeRecoderBackups', 'mqr-research') },
    name: { type: 'string', default: 'MQR research checkpoint' },
    'snapshot-id': { type: 'string' },
  },
});

async function main() {
  const operation = positionals[0] ?? 'snapshot';
  if (!['snapshot', 'status', 'verify'].includes(operation)) {
    throw new Error('Use snapshot, status, or verify. This helper cannot restore or delete code.');
  }
  const modulePath = path.join(values['coderecoder-root'], 'dist', 'backupManager.js');
  const { BackupManager } = await import(pathToFileURL(modulePath).href);
  const manager = new BackupManager();
  await manager.initialize(project, {
    storageRoot: values['storage-root'], maxBackups: 200,
    excludeNames: ['UsedCode', '.external', '.firecrawl', '__pycache__', '.pytest_cache'],
  });
  let result;
  if (operation === 'snapshot') {
    result = await manager.createBackup({
      name: values.name, prompt: 'Explicit research checkpoint via the CodeRecoder production engine',
      tags: ['protected', 'mqr-research'], skipIfUnchanged: false,
    });
    if (!result.success) throw new Error(result.error ?? result.message);
    const verification = await manager.verifyBackup(result.data.snapshot.id);
    if (!verification.success) throw new Error(verification.error ?? verification.message);
    result = { ...result, independentVerification: verification.data };
  } else if (operation === 'verify') {
    if (!values['snapshot-id']) throw new Error('verify requires --snapshot-id');
    result = await manager.verifyBackup(values['snapshot-id']);
  } else {
    result = await manager.getStatus();
  }
  console.log(JSON.stringify(result, null, 2));
  if (!result.success) process.exitCode = 1;
}

main().catch(error => {
  console.error(String(error));
  process.exitCode = 1;
});
