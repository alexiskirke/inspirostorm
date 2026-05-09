import { useState } from 'react';
import {
  AvatarCall,
  AvatarVideo,
  ControlBar,
} from '@runwayml/avatars-react';
import '@runwayml/avatars-react/styles.css';
import './App.css';

interface AvatarPersona {
  id: string;
  /** A Runway preset id (when ``isCustom`` is false) */
  presetId: string;
  /** Custom-avatar id from Runway when ``isCustom`` is true */
  customAvatarId?: string;
  displayName: string;
  accent: string;
  isCustom?: boolean;
}

const PERSONAS: Record<string, AvatarPersona> = {
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

type Status = 'idle' | 'starting' | 'live';

function getQueryParams(): URLSearchParams {
  if (typeof window === 'undefined') return new URLSearchParams();
  return new URLSearchParams(window.location.search);
}

const ACCENT_PALETTE = ['#22d3ee', '#a855f7', '#f97316', '#facc15', '#34d399', '#f472b6'];

function accentForId(id: string): string {
  let hash = 0;
  for (const c of id) hash = (hash * 31 + c.charCodeAt(0)) >>> 0;
  return ACCENT_PALETTE[hash % ACCENT_PALETTE.length];
}

function getCustomPersona(params: URLSearchParams): AvatarPersona | null {
  const customAvatarId = params.get('customAvatarId');
  if (!customAvatarId) return null;
  const displayName = params.get('name') || 'Custom Avatar';
  const accent = params.get('accent') || accentForId(customAvatarId);
  return {
    id: `custom-${customAvatarId.slice(0, 8)}`,
    presetId: '',
    customAvatarId,
    displayName,
    accent,
    isCustom: true,
  };
}

function getPersonaParam(): string | null {
  return getQueryParams().get('persona');
}

function SingleAvatar({ persona }: { persona: AvatarPersona }) {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [sessionNonce, setSessionNonce] = useState(0);

  const start = () => {
    setError(null);
    setSessionNonce((n) => n + 1);
    setStatus('starting');
    setTimeout(() => setStatus('live'), 50);
  };

  const stop = () => {
    setStatus('idle');
  };

  return (
    <div className="page page--single">
      <header className="header">
        <h1 style={{ background: persona.accent, WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent' }}>
          {persona.displayName}
        </h1>
        <p className="subtitle">
          One Runway avatar in this window. Open the <em>other</em> avatar in a second
          browser window so they can hear each other through your speakers and mic.
        </p>
        <div className="controls">
          {status === 'idle' && (
            <button className="primary" onClick={start} data-testid="start-button">
              Start {persona.displayName}
            </button>
          )}
          {status === 'starting' && (
            <button className="primary" disabled>Starting…</button>
          )}
          {status === 'live' && (
            <button className="secondary" onClick={stop} data-testid="stop-button">
              End call
            </button>
          )}
        </div>
        {error && <div className="error">{error}</div>}
      </header>

      <main>
        <section
          className="avatar-card avatar-card--full"
          style={{ borderColor: persona.accent }}
          data-testid={`avatar-card-${persona.id}`}
        >
          <header className="avatar-card__title" style={{ background: persona.accent }}>
            {persona.displayName}
          </header>
          <div className="avatar-card__stage">
            {status === 'live' ? (
              <AvatarCall
                key={`${persona.id}-${sessionNonce}`}
                avatarId={persona.isCustom ? persona.customAvatarId! : persona.presetId}
                connectUrl={
                  persona.isCustom
                    ? `/api/avatar/connect?n=${sessionNonce}-${persona.id}&customAvatarId=${encodeURIComponent(persona.customAvatarId!)}`
                    : `/api/avatar/connect?n=${sessionNonce}-${persona.id}`
                }
                video={false}
                onError={(err) => {
                  console.error(`[${persona.displayName}] error`, err);
                  setError(err instanceof Error ? err.message : String(err));
                }}
                onEnd={() => console.log(`[${persona.displayName}] call ended`)}
              >
                <AvatarVideo />
                <ControlBar />
              </AvatarCall>
            ) : (
              <div className="placeholder">
                {status === 'idle'
                  ? `Click "Start ${persona.displayName}" to connect.`
                  : 'Connecting…'}
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>
          Tip: keep this window's volume audible to your laptop mic. Browser echo
          cancellation prevents this avatar from hearing itself.
        </p>
      </footer>
    </div>
  );
}

function Launcher() {
  const openInWindow = (persona: string) => {
    const w = 720;
    const h = 820;
    const url = `/?persona=${persona}`;
    const features = `width=${w},height=${h},menubar=no,toolbar=no,location=no,status=no`;
    const offset = persona === 'chef' ? 0 : w + 40;
    window.open(
      url,
      `runway-${persona}`,
      `${features},left=${offset},top=80`,
    );
  };

  return (
    <div className="page">
      <header className="header">
        <h1>Two Runway Avatars, One Conversation</h1>
        <p className="subtitle">
          Each Runway avatar runs in its own browser window. They share your laptop's
          mic and speakers, so they hear each other like two people in the same room.
        </p>
        <p className="subtitle">
          Click each button below — your browser will pop open the two windows side by
          side. Allow mic access in <strong>both</strong>.
        </p>
        <div className="controls" style={{ flexWrap: 'wrap' }}>
          <button
            className="primary"
            onClick={() => openInWindow('chef')}
            data-testid="open-chef"
            style={{ background: PERSONAS.chef.accent }}
          >
            Open Cooking Teacher window
          </button>
          <button
            className="primary"
            onClick={() => openInWindow('designer')}
            data-testid="open-designer"
            style={{ background: PERSONAS.designer.accent }}
          >
            Open Fashion Designer window
          </button>
        </div>
        <p className="subtitle" style={{ fontSize: '0.85rem', opacity: 0.7 }}>
          Pop-up blocked? Click the popup-blocked icon in your address bar and allow
          pop-ups for this site, or open the links manually:
          <br />
          <a href="/?persona=chef" target="_blank" rel="noreferrer">
            /?persona=chef
          </a>{' '}
          •{' '}
          <a href="/?persona=designer" target="_blank" rel="noreferrer">
            /?persona=designer
          </a>
        </p>
      </header>
    </div>
  );
}

function App() {
  const params = getQueryParams();
  const customPersona = getCustomPersona(params);
  if (customPersona) {
    return <SingleAvatar persona={customPersona} />;
  }
  const personaKey = params.get('persona');
  const persona = personaKey ? PERSONAS[personaKey] : null;
  if (persona) {
    return <SingleAvatar persona={persona} />;
  }
  return <Launcher />;
}

export default App;
