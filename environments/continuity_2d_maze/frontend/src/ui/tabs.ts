// iter D5: Q1/Q2 tab state for the UI. The active tab is mirrored into the
// URL hash (`#q1` / `#q2`) so reloads and deep links preserve context, and
// so Playwright can still trigger Q2 with a hash-based URL if it wants to.
// Hash changes use history.replaceState to avoid clobbering back/forward.

export type TabKey = 'q1' | 'q2';

const TAB_KEYS: readonly TabKey[] = ['q1', 'q2'];

type TabChangeHandler = (tab: TabKey) => void;
const handlers: TabChangeHandler[] = [];

let current: TabKey = readTabFromHash();

function readTabFromHash(): TabKey {
  if (typeof location === 'undefined') return 'q1';
  return location.hash === '#q2' ? 'q2' : 'q1';
}

export function getActiveTab(): TabKey {
  return current;
}

export function setActiveTab(tab: TabKey, options: { updateHash?: boolean } = {}): void {
  if (!TAB_KEYS.includes(tab) || tab === current) return;
  current = tab;
  syncDomClasses(tab);
  if (options.updateHash !== false && typeof history !== 'undefined') {
    history.replaceState(null, '', `#${tab}`);
  }
  for (const h of handlers) h(tab);
}

export function onTabChange(handler: TabChangeHandler): void {
  handlers.push(handler);
}

function syncDomClasses(tab: TabKey): void {
  document
    .querySelectorAll('button.tab')
    .forEach((el) => el.classList.toggle('active', (el as HTMLElement).dataset.tab === tab));
  document
    .querySelectorAll('.tab-panel')
    .forEach((el) => {
      const active = (el as HTMLElement).dataset.tab === tab;
      (el as HTMLElement).style.display = active ? '' : 'none';
    });
}

export function initTabUI(): void {
  // Bind tab buttons and seed the DOM with the initial active tab.
  syncDomClasses(current);
  document.querySelectorAll('button.tab').forEach((el) => {
    const key = (el as HTMLElement).dataset.tab as TabKey | undefined;
    if (!key || !TAB_KEYS.includes(key)) return;
    el.addEventListener('click', () => setActiveTab(key));
  });
  window.addEventListener('hashchange', () => {
    const next = readTabFromHash();
    if (next !== current) setActiveTab(next, { updateHash: false });
  });
}
