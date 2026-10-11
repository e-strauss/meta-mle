# ML search dashboard

A small Next.js dashboard for ML experiment harness runs: which runs are live,
past runs, the search tree with every node's proposal, code, repairs, scores and
findings, and a form to start new runs. It is independent of any harness: it
reads run directories strictly read-only and never imports harness code.
Nothing in meta-mle depends on it.

```
dashboard/
  launchers.json        runs roots + launch templates (committed defaults)
  launchers.local.json  optional machine-specific overrides (gitignored)
  lib/types.ts          harness-neutral run model (runs, nodes, edges, sections)
  lib/adapters/         one adapter per harness; meta-mle.ts is the only code
                        that knows meta-mle's state.db / artifacts layout
  lib/runs.ts           discovers runs under the roots, guarded file access
  lib/launch.ts         validates a launch form, starts scripts/launch-runner.mjs
  lib/procs.ts          liveness from /proc/locks (never takes a lock)
  components/           tree view, score chart, node panel, launches
  app/                  runs list, run page, new-run page, login
```

## Setup (once)

Node 24+ is required (the adapter uses the built-in `node:sqlite`).

```bash
cd dashboard
npm ci
npm run set-password      # single login, written to .env.local (mode 0600)
```

The dashboard needs no harness credentials. Launched processes inherit its
environment minus its own login secrets; harness configuration (model API keys,
`GOOGLE_APPLICATION_CREDENTIALS` for remote data) belongs in the harness's own
environment, e.g. meta-mle's repository `.env`.

## Run

```bash
dashboard/run.sh          # development server on 127.0.0.1:3200
dashboard/run.sh prod     # production build, then serve it
```

From a laptop: `ssh -N -L 3200:localhost:3200 <node>` and open
<http://localhost:3200>. `PORT=3300 dashboard/run.sh` changes the port.

## How it works

- **Theme.** The header's Theme selector offers System, Light and Dark. The choice
  is saved in a browser cookie and applied before the page renders. The bottom-left
  Next.js menu controls only the development tools' appearance.
- **Runs.** Every directory under a runs root that an adapter recognises is a run;
  its id is `<root name>/<path below root>`. A run is *live* when a process holds
  its lock file (read from `/proc/locks`); the model is read from that process's
  command line, or from the launch record for dashboard launches. Live pages
  refresh every 5 seconds.
- **Search tree.** Candidates hang off their selected parent; grid siblings from one
  expansion are boxed together. Explorations and setups attach to where they were
  requested; dashed edges show evidence used by an expansion and MCGS references.
  Colour encodes score (low → high), with ✕ failed, ⋯ running and a ring for the best.
  Click a node for its proposal, configuration, fold scores, the full repair chain
  (plan, error, artifacts) and the findings it could see.
- **Launching.** A launcher is a list of argv templates plus typed form fields
  (`slug`, `text`, `int`, `select`, `file`). Values are validated against the field
  definitions, substituted into the templates and executed without a shell by a
  detached `scripts/launch-runner.mjs`, which records `.launches/<id>/` (command,
  output log, status). Launches survive dashboard restarts and can be stopped with
  SIGINT, which meta-mle records as an interruption.

## Adding another harness

Implement `Adapter` from `lib/types.ts` (`detect`, `summary`, `detail`, `node`) in
`lib/adapters/<harness>.ts`, register it in `lib/adapters/index.ts`, and add a
launcher to `launchers.json` or `launchers.local.json`.
