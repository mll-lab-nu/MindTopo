import { Pair } from '../maze/types';
import { TabKey } from './tabs';

export type PairSelectHandler = (pairIndex: number | null) => void;

function pairListId(tab: TabKey): string {
  return tab === 'q2' ? 'q2-pair-list' : 'q1-pair-list';
}

export function renderPairListDOM(
  tab: TabKey,
  pairs: Pair[],
  onSelect: PairSelectHandler,
): void {
  const list = document.getElementById(pairListId(tab));
  if (!list) return;
  list.innerHTML = '';
  pairs.forEach((p, i) => {
    const li = document.createElement('li');
    li.dataset.index = String(i);
    li.innerHTML =
      `<span class="names">${p.a_name}–${p.b_name}</span>` +
      `<span class="verdict ${p.connected ? 'yes' : 'no'}">${p.connected ? 'yes' : 'no'}</span>`;
    li.addEventListener('click', () => {
      const already = li.classList.contains('selected');
      list.querySelectorAll('li.selected').forEach((el) => el.classList.remove('selected'));
      if (already) {
        onSelect(null);
      } else {
        li.classList.add('selected');
        onSelect(i);
      }
    });
    list.appendChild(li);
  });
}

export function clearPairSelectionDOM(tab: TabKey): void {
  document
    .getElementById(pairListId(tab))
    ?.querySelectorAll('li.selected')
    .forEach((el) => el.classList.remove('selected'));
}
