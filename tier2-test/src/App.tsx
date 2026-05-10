/**
 * Tier-2 concurrency testbench.
 *
 * Three modes, selected by URL params:
 *
 *   /                           -> Launcher: open both avatars in two
 *                                  separate browser windows so each
 *                                  starts its own RTC session in its own
 *                                  process. This was the workaround on
 *                                  tier 1 — re-running it on tier 2
 *                                  should still work and gives a
 *                                  baseline.
 *
 *   /?avatar=cooking-teacher    -> SingleAvatar: one preset avatar in
 *                                  this window. Used by the launcher.
 *
 *   /?mode=both                 -> BothAvatars: both presets running
 *                                  side-by-side in the SAME window. This
 *                                  is the real tier-2 stress test — on
 *                                  tier 1 the second session would queue
 *                                  behind the first.
 *
 * Each AvatarCall fetches its session through ``/api/avatar/connect``;
 * we tag the request with a per-instance nonce in the URL so that the
 * Runway React SDK's react-query cache can't collapse the two calls
 * into one.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AvatarCall,
  AvatarVideo,
  ControlBar,
} from '@runwayml/avatars-react';
import '@runwayml/avatars-react/styles.css';

interface Persona {
  id: 'chef' | 'designer';
  presetId: string;
  displayName: string;
  accent: string;
}

const PERSONAS: Record<Persona['id'], Persona> = {
  chef: {
    id: 'chef',
    presetId: 'cooking-teacher',
    displayName: 'Cooking Teacher',
    accent: '#ff8c42',
  },
  designer: {
    id: 'designer',
    presetId: 'fashion-designer',
    displayName: 'Fashion Designer',
    accent: '#a855f7',
  },
};

type Status = 'idle' | 'starting' | 'live' | 'error';

interface EventRow {
  ts: string;
  who: string;
  msg: string;
  err?: boolean;
}

function getQuery(): URLSearchParams {
  if (typeof window === 'undefined') return new URLSearchParams();
  return new URLSearchParams(window.location.search);
}

function ts() {
  const d = new Date();
  return d.toTimeString().slice(0, 8) + '.' + String(d.getMilliseconds()).padStart(3, '0');
}

/** A single avatar tile. Owns its own status + a per-tile nonce so each
 *  reconnect is a fresh session (avoids react-query cache reuse). */
function AvatarTile({
  persona,
  onEvent,
}: {
  persona: Persona;
  onEvent?: (e: EventRow) => void;
}) {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const log = (msg: string, err = false) => {
    onEvent?.({ ts: ts(), who: persona.displayName, msg, err });
    if (err) console.error(`[${persona.displayName}]`, msg);
    else console.log(`[${persona.displayName}]`, msg);
  };

  const start = () => {
    setError(null);
    setNonce((n) => n + 1);
    setStatus('starting');
    log('start clicked');
    setTimeout(() => setStatus('live'), 50);
  };
  const stop = () => {
    setStatus('idle');
    log('end clicked');
  };

  return (
    <div className="card" data-testid={`tile-${persona.id}`}>
      <div className="card__title" style={{ background: persona.accent }}>
        {persona.displayName}
      </div>
      <div className="card__stage">
        {status === 'live' ? (
          <AvatarCall
            key={`${persona.id}-${nonce}`}
            avatarId={persona.presetId}
            connectUrl={`/api/avatar/connect?n=${nonce}-${persona.id}`}
            video={false}
            onError={(err) => {
              const msg = err instanceof Error ? err.message : String(err);
              setStatus('error');
              setError(msg);
              log(`error: ${msg}`, true);
            }}
            onEnd={() => {
              log('call ended');
              if (status === 'live') setStatus('idle');
            }}
          >
            <AvatarVideo />
            <ControlBar />
          </AvatarCall>
        ) : (
          <div className="card__placeholder">
            {status === 'idle' && `Press Start to connect ${persona.displayName}.`}
            {status === 'starting' && `Connecting ${persona.displayName}…`}
            {status === 'error' && (error || 'Failed.')}
          </div>
        )}
      </div>
      <div className="card__footer">
        <span className={`tag ${
          status === 'live' ? 'tag--ok' :
          status === 'error' ? 'tag--err' :
          status === 'starting' ? 'tag--warn' :
          'tag--idle'
        }`}>{status}</span>
        <div className="controls" style={{ margin: 0 }}>
          {status === 'live' || status === 'error' ? (
            <button className="secondary" onClick={stop}>Stop</button>
          ) : (
            <button
              className="primary"
              onClick={start}
              disabled={status === 'starting'}
              data-testid={`start-${persona.id}`}
            >
              {status === 'starting' ? 'Starting…' : `Start ${persona.displayName}`}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function EventsLog({ events }: { events: EventRow[] }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [events.length]);
  if (events.length === 0) return null;
  return (
    <div className="events" ref={ref}>
      {events.map((e, i) => (
        <div key={i} className={`events__row${e.err ? ' err' : ''}`}>
          <span>{e.ts}</span> · <b>{e.who}</b> · {e.msg}
        </div>
      ))}
    </div>
  );
}

function SingleAvatar({ persona }: { persona: Persona }) {
  const [events, setEvents] = useState<EventRow[]>([]);
  const push = (e: EventRow) => setEvents((prev) => [...prev, e]);
  return (
    <div className="shell">
      <h1>{persona.displayName}</h1>
      <p className="subtitle">
        One preset avatar in its own browser window. The other window has
        the second avatar; their audio reaches each other through the
        laptop's speakers and mic. Browser echo cancellation keeps each
        avatar from hearing itself.
      </p>
      <div className="stage" style={{ gridTemplateColumns: '1fr' }}>
        <AvatarTile persona={persona} onEvent={push} />
      </div>
      <EventsLog events={events} />
    </div>
  );
}

function BothAvatars() {
  const [events, setEvents] = useState<EventRow[]>([]);
  const push = (e: EventRow) => setEvents((prev) => [...prev, e]);
  return (
    <div className="shell">
      <h1>Both Runway Avatars · Same Window</h1>
      <p className="subtitle">
        Both preset avatars rendered in the SAME tab. On tier 1 only one
        could provision; the second would queue until the first ended.
        On tier 2 we expect both to reach <code>READY</code> in parallel.
        Click <em>Start</em> on each, then watch the event log below.
      </p>
      <div className="stage">
        <AvatarTile persona={PERSONAS.chef} onEvent={push} />
        <AvatarTile persona={PERSONAS.designer} onEvent={push} />
      </div>
      <EventsLog events={events} />
    </div>
  );
}

function Launcher() {
  const open = (id: Persona['id']) => {
    const w = 720;
    const h = 820;
    const url = `/?avatar=${PERSONAS[id].presetId}`;
    const offsetLeft = id === 'chef' ? 40 : w + 80;
    window.open(
      url,
      `tier2-${id}`,
      `width=${w},height=${h},menubar=no,toolbar=no,left=${offsetLeft},top=80`,
    );
  };
  return (
    <div className="shell">
      <h1>Runway Tier-2 Concurrency Test</h1>
      <p className="subtitle">
        Two ways to test that two preset avatars can run at the same time.
        The same-window test is the real measure; the two-window option is
        a sanity baseline (it should always work, since each window is its
        own JS context and its own RTC session).
      </p>
      <div className="controls">
        <a className="button" href="/?mode=both" data-testid="open-both">
          ▶ Both avatars in this window
        </a>
        <button
          className="primary"
          onClick={() => {
            open('chef');
            open('designer');
          }}
          style={{ background: 'linear-gradient(90deg, #ff8c42, #a855f7)' }}
          data-testid="open-windows"
        >
          ⧉ Open in two separate windows
        </button>
      </div>
      <p className="subtitle muted" style={{ fontSize: '0.85rem' }}>
        Pop-ups blocked? Click the icon in your address bar and allow
        pop-ups for <code>localhost:5174</code>, or open the links manually:{' '}
        <a href="/?avatar=cooking-teacher" target="_blank" rel="noreferrer">/?avatar=cooking-teacher</a>{' '}·{' '}
        <a href="/?avatar=fashion-designer" target="_blank" rel="noreferrer">/?avatar=fashion-designer</a>.
      </p>
    </div>
  );
}

export default function App() {
  const params = useMemo(getQuery, []);
  const mode = params.get('mode');
  const avatarPresetId = params.get('avatar');

  if (mode === 'both') return <BothAvatars />;
  if (avatarPresetId) {
    const persona = Object.values(PERSONAS).find((p) => p.presetId === avatarPresetId);
    if (persona) return <SingleAvatar persona={persona} />;
    // Unknown preset — fall through to launcher with a hint.
    return (
      <div className="shell">
        <h1>Unknown avatar preset</h1>
        <p className="error">No preset matches <code>{avatarPresetId}</code>. Try one of:{' '}
          {Object.values(PERSONAS).map((p) => p.presetId).join(', ')}.</p>
      </div>
    );
  }
  return <Launcher />;
}
