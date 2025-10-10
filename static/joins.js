// public/joins.js
;(() => {
  // Global project state (kept tiny; no framework)
  window.PROJECT = window.PROJECT || {
    mainTable: null,
    joins: [],      // [{ type: 'LEFT', table: 'inventory_items', key: 'product_id' }]
    tablesList: [   // TODO: keep in sync with backend registry
      'order_items',
      'inventory_items',
      'orders',
      'products',
      'customers'
    ]
  };

  // Attach handlers after DOM is ready
  document.addEventListener('DOMContentLoaded', () => {
    const dd = document.querySelector('#project-table-select');
    const joinBtn = document.querySelector('#btn-join');
    const joinHost = document.querySelector('#join-pills');

    if (dd) {
      window.PROJECT.mainTable = dd.value || null;
      dd.addEventListener('change', () => {
        window.PROJECT.mainTable = dd.value || null;
      });
    }

    if (joinBtn) {
      joinBtn.addEventListener('click', openJoinModal);
    }

    renderJoinPills(joinHost);
  });

  function renderJoinPills(host) {
    if (!host) return;
    host.innerHTML = '';
    if (!window.PROJECT.joins.length) return;
    window.PROJECT.joins.forEach((j, idx) => {
      const pill = document.createElement('div');
      pill.className = 'join-pill';
      pill.innerHTML = `
        <span class="join-pill-text">${j.type} JOIN ${j.table} ON ${j.key}</span>
        <button class="join-pill-remove" title="Remove">✕</button>
      `;
      pill.querySelector('.join-pill-remove').addEventListener('click', () => {
        window.PROJECT.joins.splice(idx, 1);
        renderJoinPills(host);
      });
      host.appendChild(pill);
    });
  }

  function openJoinModal() {
    const modal = document.createElement('div');
    modal.className = 'join-modal-backdrop';
    modal.innerHTML = `
      <div class="join-modal">
        <div class="join-modal-title">Add Join</div>
        <label class="join-modal-row">
          <span>Join type</span>
          <select id="join-type">
            <option>INNER</option>
            <option>LEFT</option>
            <option>RIGHT</option>
            <option>FULL</option>
          </select>
        </label>
        <label class="join-modal-row">
          <span>Table</span>
          <select id="join-table">
            ${window.PROJECT.tablesList.map(t => `<option>${t}</option>`).join('')}
          </select>
        </label>
        <label class="join-modal-row">
          <span>Key (single column)</span>
          <input id="join-key" placeholder="e.g., product_id" />
        </label>
        <div class="join-modal-actions">
          <button id="join-cancel">Cancel</button>
          <button id="join-save">Add</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    modal.querySelector('#join-cancel').onclick = () => modal.remove();
    modal.querySelector('#join-save').onclick = () => {
      const type = modal.querySelector('#join-type').value.trim().toUpperCase();
      const table = modal.querySelector('#join-table').value.trim();
      const key = modal.querySelector('#join-key').value.trim();

      if (!type || !table || !key) { alert('Please fill all fields.'); return; }

      window.PROJECT.joins.push({ type, table, key });
