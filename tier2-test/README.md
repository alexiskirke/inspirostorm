# tier2-test

Standalone testbench for verifying that two RunwayML preset avatars can run
concurrently on tier 2 (the upgraded plan). Lives in its own dir on its own
ports so it can run side-by-side with the main `app/` and `scout/` projects
without interference.

- Vite client → http://localhost:5174
- Express server → http://localhost:3002 (proxied as `/api/*` from the client)

## Run

From this directory:

```bash
npm install        # one time
npm run dev        # starts vite + express together
```

Then open http://localhost:5174.

## Modes

The landing page offers two test paths:

1. **Both avatars in this window** (`/?mode=both`) — the real concurrency
   test. Two `<AvatarCall>` components on the same page, each making a
   separate `realtime_sessions.create` call. On tier 1 this would queue
   the second avatar until the first ended; on tier 2 they should both
   reach `READY` in parallel.

2. **Open in two separate windows** — pops two browser windows, one per
   preset avatar. Each window has its own JS context and its own RTC
   session, so this is the easiest baseline. Worked even on tier 1.

The page also shows a per-avatar event log so you can see in the UI
whether both avatars actually went `live` simultaneously, and the express
server prints timing-tagged logs (e.g. `[r1] +874ms status=READY`) so you
can compare the two sessions' provisioning times side-by-side.
