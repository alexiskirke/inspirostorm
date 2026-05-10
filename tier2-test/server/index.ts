/**
 * Tier-2 testbench server.
 *
 * Single endpoint: POST /api/avatar/connect — accepts an avatarId
 * (a Runway preset id), creates a realtime session, polls until READY
 * (or fails), and returns the credentials the React SDK consumes.
 *
 * Logs are deliberately verbose and tagged with a per-request id so we
 * can see in the console whether two simultaneous sessions actually
 * provision in parallel (tier 2 expectation) or if one queues behind
 * the other (tier 1 behaviour).
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express, { type Request, type Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Runway from '@runwayml/sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// We honour both this dir's .env and the workspace-root .env (the user
// keeps RUNWAYML_API_KEY at the top level).
dotenv.config({ path: path.join(__dirname, '..', '.env') });
dotenv.config({ path: path.join(__dirname, '..', '..', '.env') });

const apiKey = process.env.RUNWAYML_API_SECRET || process.env.RUNWAYML_API_KEY;
if (!apiKey) {
  console.error('[tier2] Missing RUNWAYML_API_KEY / RUNWAYML_API_SECRET in env.');
  process.exit(1);
}

const runway = new Runway({ apiKey });
const app = express();
app.use(cors());
app.use(express.json());

interface ConnectBody {
  avatarId?: string;
  startScript?: string;
  personality?: string;
}

let reqCounter = 0;

app.post(
  '/api/avatar/connect',
  async (req: Request<unknown, unknown, ConnectBody>, res: Response) => {
    const reqId = `r${++reqCounter}`;
    const t0 = Date.now();
    const { avatarId, startScript, personality } = req.body || {};
    const nonce = String(req.query.n || '');

    if (!avatarId) {
      console.warn(`[${reqId}] missing avatarId`);
      return res.status(400).json({ error: 'avatarId is required' });
    }

    console.log(
      `[${reqId}] +0ms POST /connect avatar=${avatarId} nonce=${nonce}`,
    );

    try {
      const sessionPayload: Record<string, unknown> = {
        model: 'gwm1_avatars',
        avatar: { type: 'runway-preset', presetId: avatarId },
      };
      if (startScript) sessionPayload.startScript = startScript;
      if (personality) sessionPayload.personality = personality;

      const created = await runway.realtimeSessions.create(sessionPayload as never);
      const sessionId = (created as { id: string }).id;
      console.log(
        `[${reqId}] +${Date.now() - t0}ms sessionCreated id=${sessionId}`,
      );

      const deadline = Date.now() + 120_000;
      let lastStatus = '';
      while (Date.now() < deadline) {
        const session = (await runway.realtimeSessions.retrieve(sessionId)) as {
          status: string;
          sessionKey?: string;
          failure?: unknown;
          queued?: boolean;
        };
        if (session.status !== lastStatus) {
          console.log(
            `[${reqId}] +${Date.now() - t0}ms ${sessionId} status=${session.status}` +
              (session.queued ? ' (QUEUED — tier limit?)' : ''),
          );
          lastStatus = session.status;
        }
        if (session.status === 'READY') {
          console.log(
            `[${reqId}] +${Date.now() - t0}ms READY — returning sessionKey`,
          );
          return res.json({ sessionId, sessionKey: session.sessionKey });
        }
        if (session.status === 'FAILED' || session.status === 'CANCELLED') {
          console.warn(
            `[${reqId}] +${Date.now() - t0}ms ${session.status}`,
          );
          return res.status(502).json({
            error: `Session ${session.status}`,
            failure: session.failure,
          });
        }
        await new Promise((r) => setTimeout(r, 1_000));
      }
      console.warn(
        `[${reqId}] +${Date.now() - t0}ms TIMEOUT (last status: ${lastStatus})`,
      );
      return res.status(504).json({ error: 'Session creation timed out' });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`[${reqId}] +${Date.now() - t0}ms exception:`, message);
      return res.status(500).json({ error: message });
    }
  },
);

app.get('/api/health', (_req, res) => res.json({ ok: true }));

const port = Number(process.env.PORT) || 3002;
app.listen(port, () => {
  console.log(`[tier2] server listening on http://localhost:${port}`);
});
