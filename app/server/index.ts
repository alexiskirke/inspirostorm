import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express, { type Request, type Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Runway from '@runwayml/sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Try app/.env then ../.env (the user keeps their key in the workspace root)
dotenv.config({ path: path.join(__dirname, '..', '.env') });
dotenv.config({ path: path.join(__dirname, '..', '..', '.env') });

const apiKey = process.env.RUNWAYML_API_SECRET || process.env.RUNWAYML_API_KEY;

if (!apiKey) {
  console.error('Missing RUNWAYML_API_KEY / RUNWAYML_API_SECRET in environment.');
  process.exit(1);
}

const runway = new Runway({ apiKey });

const app = express();
app.use(cors());
app.use(express.json());

interface ConnectBody {
  avatarId?: string;
  customAvatarId?: string;
  startScript?: string;
  personality?: string;
}

app.post('/api/avatar/connect', async (req: Request<unknown, unknown, ConnectBody>, res: Response) => {
  const body = req.body || {};
  // The Runway React SDK only sends `avatarId` in the body. To trigger
  // custom-avatar mode we therefore also honour `?customAvatarId=...` on
  // the URL — the React side just appends it to `connectUrl`.
  const queryCustom = typeof req.query.customAvatarId === 'string' ? req.query.customAvatarId : undefined;
  const customAvatarId = body.customAvatarId || queryCustom;
  const avatarId = customAvatarId ? undefined : body.avatarId;
  const startScript = body.startScript;
  const personality = body.personality;

  if (!avatarId && !customAvatarId) {
    return res.status(400).json({ error: 'avatarId or customAvatarId is required' });
  }

  try {
    const avatarConfig = customAvatarId
      ? { type: 'custom', avatarId: customAvatarId }
      : { type: 'runway-preset', presetId: avatarId };
    const sessionPayload: Record<string, unknown> = {
      model: 'gwm1_avatars',
      avatar: avatarConfig,
    };
    // Personality / startScript only override the avatar's defaults; for
    // custom avatars Runway already stores the persona we set at create
    // time, so we deliberately skip these unless explicitly provided.
    if (startScript) sessionPayload.startScript = startScript;
    if (personality) sessionPayload.personality = personality;

    const created = await runway.realtimeSessions.create(sessionPayload as never);
    const sessionId = (created as { id: string }).id;
    console.log(
      `[connect] created session ${sessionId} for ${customAvatarId ? `custom:${customAvatarId}` : `preset:${avatarId}`}`,
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
          `[connect] ${sessionId} status=${session.status}` +
            (session.queued ? ' (queued)' : ''),
        );
        lastStatus = session.status;
      }
      if (session.status === 'READY') {
        return res.json({ sessionId, sessionKey: session.sessionKey });
      }
      if (session.status === 'FAILED' || session.status === 'CANCELLED') {
        return res.status(502).json({
          error: `Session ${session.status}`,
          failure: session.failure,
        });
      }
      await new Promise((resolve) => setTimeout(resolve, 1_000));
    }
    console.log(`[connect] ${sessionId} timed out (last status: ${lastStatus})`);
    return res.status(504).json({ error: 'Session creation timed out' });
  } catch (err) {
    console.error('Failed to create avatar session:', err);
    const message = err instanceof Error ? err.message : String(err);
    return res.status(500).json({ error: message });
  }
});

app.get('/api/health', (_req, res) => {
  res.json({ ok: true });
});

const distDir = path.join(__dirname, '..', 'dist');
app.use(express.static(distDir));
app.get(/^\/(?!api).*/, (_req, res) => {
  res.sendFile(path.join(distDir, 'index.html'));
});

const port = Number(process.env.PORT) || 3001;
app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
