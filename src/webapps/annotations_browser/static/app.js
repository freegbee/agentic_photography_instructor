const state = {
  page: 1,
  pageSize: 20,
  sort: 'file_name',
  order: 'asc',
  category: '',
  annotation: ''
}

async function fetchCategories() {
  try {
    let r = await fetch('/api/categories');
    let j = await r.json();
    const sel = document.getElementById('categorySelect');
    sel.innerHTML = '<option value="">All categories</option>';
    j.categories.forEach(c => {
      const o = document.createElement('option');
      o.value = c.name;
      o.textContent = c.name;
      sel.appendChild(o);
    });
  } catch (e) {
    console.error('Failed to load categories', e);
  }
}

async function fetchAnnotationFiles(){
  try{
    let r = await fetch('/api/list_annotation_files');
    let j = await r.json();
    const sel = document.getElementById('fileSelect');
    sel.innerHTML = '<option value="">Select annotations.json...</option>';
    j.files.forEach(f=>{
      const o = document.createElement('option');
      o.value = f;
      o.textContent = f;
      sel.appendChild(o);
    });
  }catch(e){console.error('Failed to list annotation files', e)}
}

async function loadImages() {
  const grid = document.getElementById('grid');
  grid.innerHTML = 'Loading...';
  const params = new URLSearchParams();
  if (state.category) params.set('category', state.category);
  if (state.annotation) params.set('annotation', state.annotation);
  params.set('sort', state.sort);
  params.set('order', state.order);
  params.set('page', String(state.page));
  params.set('page_size', String(state.pageSize));

  try {
    let r = await fetch('/api/images?' + params.toString());
    let j = await r.json();
    renderGrid(j);
  } catch (e) {
    grid.innerHTML = 'Error loading images';
    console.error(e);
  }
}

function renderGrid(data) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  document.getElementById('pageInfo').textContent = `Page ${data.page} of ${Math.ceil(data.total / data.page_size || 1)}`;

  data.items.forEach(item => {
    const el = document.createElement('div');
    el.className = 'thumb';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = item.thumbnail_url;
    img.alt = item.file_name;
    img.addEventListener('click', () => openModal(item.image_url));
    // show filename only as tooltip (title) and aria-label on the image
    img.title = item.file_name;
    img.setAttribute('aria-label', item.file_name);
    img.setAttribute('role', 'img');
    el.appendChild(img);

    // show scores/categories info if available
    const meta = item.meta || {};

    // helper to format numbers to 4 decimal places (shared)
    const fmt = (v) => {
      if (v === undefined || v === null || isNaN(Number(v))) return '-';
      try { return Number(v).toFixed(4); } catch(e) { return '-'; }
    }

    // Categories table (exclude category_id == 0)
    // Prefer detailed per-category meta if available (meta.categories_meta),
    // otherwise fall back to meta.categories and show '-' for change.
    const catsMeta = Array.isArray(meta.categories_meta) ? meta.categories_meta.filter(c => c && c.id !== 0) : null;
    const catsBasic = Array.isArray(meta.categories) ? meta.categories.filter(c => c && c.id !== 0) : [];
    const catsToRender = catsMeta !== null ? catsMeta : catsBasic.map(c => ({ id: c.id, name: c.name, change: null }));
    if (catsToRender.length > 0) {
      const catTbl = document.createElement('table');
      catTbl.className = 'categories-table';
      const catTbody = document.createElement('tbody');
      catsToRender.forEach(c => {
        const tr = document.createElement('tr');
        const tdName = document.createElement('td');
        tdName.className = 'label';
        tdName.textContent = c.name || String(c.id);
        const tdChange = document.createElement('td');
        tdChange.className = 'num';
        // use per-category change if available, otherwise '-' (fmt handles null)
        const changeVal = (c.change !== undefined) ? c.change : null;
        tdChange.textContent = fmt(changeVal);
        tr.appendChild(tdName);
        tr.appendChild(tdChange);
        catTbody.appendChild(tr);
      });
      catTbl.appendChild(catTbody);
      el.appendChild(catTbl);

      // spacer between categories and scores
      const spacer = document.createElement('div');
      spacer.className = 'table-spacer';
      el.appendChild(spacer);
    }

    // show scores table (below categories)
    if (meta.score !== undefined || meta.initial_score !== undefined || meta.change !== undefined) {
      // create a compact 3-row table: label (left) | value (right)
      const tbl = document.createElement('table');
      tbl.className = 'score-table';
      const tbody = document.createElement('tbody');

      const rows = [
        ['Score', meta.score],
        ['Initial', meta.initial_score],
        ['Change', meta.change]
      ];

      rows.forEach(rw => {
        const tr = document.createElement('tr');
        const tdLabel = document.createElement('td');
        tdLabel.className = 'label';
        tdLabel.textContent = rw[0];
        const tdVal = document.createElement('td');
        tdVal.className = 'num';
        tdVal.textContent = fmt(rw[1]);
        tr.appendChild(tdLabel);
        tr.appendChild(tdVal);
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody);
      el.appendChild(tbl);
    }
    grid.appendChild(el);
  });
}

function openModal(imageUrl) {
  const modal = document.getElementById('modal');
  const img = document.getElementById('modalImage');
  img.src = imageUrl;
  modal.classList.remove('hidden');
}

function closeModal() {
  const modal = document.getElementById('modal');
  const img = document.getElementById('modalImage');
  img.src = '';
  modal.classList.add('hidden');
}

let _loadStatusInterval = null;
async function _fetchLoadStatus() {
  try {
    const r = await fetch('/api/load_status');
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

function showSpinner() {
  const gp = document.getElementById('glasspane');
  const msg = document.getElementById('glasspaneMsg');
  const counts = document.getElementById('glasspaneCounts');
  const loadBtn = document.getElementById('loadFileBtn');
  if (!gp) return;
  gp.classList.remove('hidden');
  gp.setAttribute('aria-hidden', 'false');
  if (loadBtn) loadBtn.disabled = true;

  // initial message
  if (msg) msg.textContent = 'Loading annotations...';
  if (counts) counts.textContent = '';

  // wire cancel button
  const cancelBtn = document.getElementById('glasspaneCancelBtn');
  if (cancelBtn) {
    cancelBtn.disabled = false;
    cancelBtn.onclick = async () => {
      cancelBtn.disabled = true;
      if (msg) msg.textContent = 'Cancelling...';
      try {
        const r = await fetch('/api/load_cancel', {method: 'POST'});
        if (!r.ok) {
          const text = await r.text();
          console.warn('cancel failed', r.status, text);
        }
      } catch (e) {
        console.warn('cancel request failed', e);
      }
    }
  }

  // start polling status every 800ms
  if (_loadStatusInterval) clearInterval(_loadStatusInterval);
  _loadStatusInterval = setInterval(async () => {
    const st = await _fetchLoadStatus();
    if (!st) return;
    if (msg) msg.textContent = `State: ${st.state} — file: ${st.file || ''}`;
    if (counts) counts.textContent = st.total ? `Parsed: ${st.processed}/${st.total} images` : `Parsed: ${st.processed} images`;
    if (st.state === 'done' || st.state === 'error' || st.state === 'cancelled') {
      // stop polling shortly after completion to allow UI update
      clearInterval(_loadStatusInterval);
      _loadStatusInterval = null;
      if (loadBtn) loadBtn.disabled = false;
      if (cancelBtn) cancelBtn.disabled = true;

      // show skipped images modal if present
      if (st.skipped_images && st.skipped_images.length > 0) {
        setTimeout(() => showSkippedModal(st.skipped_images), 100);
      }
    }
  }, 800);
}

function hideSpinner() {
  const gp = document.getElementById('glasspane');
  const loadBtn = document.getElementById('loadFileBtn');
  if (!gp) return;
  gp.classList.add('hidden');
  gp.setAttribute('aria-hidden', 'true');
  if (_loadStatusInterval) {
    clearInterval(_loadStatusInterval);
    _loadStatusInterval = null;
  }
  if (loadBtn) loadBtn.disabled = false;
  const msg = document.getElementById('glasspaneMsg');
  const counts = document.getElementById('glasspaneCounts');
  if (msg) msg.textContent = '';
  if (counts) counts.textContent = '';
}

function showSkippedModal(skipped) {
  const modal = document.getElementById('skippedModal');
  const list = document.getElementById('skippedList');
  if (!modal || !list) return;
  list.innerHTML = '';
  if (!skipped || skipped.length === 0) return;
  skipped.forEach(s => {
    const li = document.createElement('li');
    li.textContent = `${s.file_name || s.file || ''} — ${s.reason || ''}`;
    list.appendChild(li);
  });
  modal.classList.remove('hidden');
}

function hideSkippedModal() {
  const modal = document.getElementById('skippedModal');
  if (!modal) return;
  modal.classList.add('hidden');
}

// wire controls
window.addEventListener('load', async () => {
  await fetchAnnotationFiles();
  // do not auto-load categories/images until user confirms a file
  document.getElementById('applyBtn').addEventListener('click', () => {
    state.category = document.getElementById('categorySelect').value;
    state.annotation = document.getElementById('annotationInput').value;
    state.sort = document.getElementById('sortSelect').value;
    state.order = document.getElementById('orderSelect').value;
    state.page = 1;
    state.pageSize = parseInt(document.getElementById('pageSize').value) || 20;
    if (state.pageSize > 100) state.pageSize = 100;
    loadImages();
  });
  document.getElementById('prevBtn').addEventListener('click', () => {
    if (state.page > 1) {
      state.page -= 1;
      loadImages();
    }
  });
  document.getElementById('nextBtn').addEventListener('click', () => {
    state.page += 1;
    loadImages();
  });
  // page size input: apply immediately on change (bound to max 100)
  const pageSizeInput = document.getElementById('pageSize');
  if (pageSizeInput) {
    pageSizeInput.addEventListener('change', () => {
      let v = parseInt(pageSizeInput.value, 10) || 20;
      if (v < 1) v = 1;
      if (v > 100) v = 100;
      pageSizeInput.value = v;
      state.pageSize = v;
      state.page = 1; // reset to first page when page size changes
      loadImages();
    });
  }
  document.getElementById('loadFileBtn').addEventListener('click', async () => {
    const sel = document.getElementById('fileSelect');
    const fp = sel.value;
    if (!fp) return alert('Please select an annotations.json file');
    try{
      showSpinner();
      const r = await fetch('/api/load_annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_path: fp})});
      if(!r.ok) throw new Error('Load failed');
      // after starting background load, poll until done, then refresh categories/images
      // we'll wait until load_status reports done
      const waitForDone = async () => {
        while (true) {
          const st = await _fetchLoadStatus();
          if (!st) break;
          if (st.state === 'done' || st.state === 'cancelled') return st;
          if (st.state === 'error') throw new Error(st.error || 'load error');
          await new Promise(r => setTimeout(r, 500));
        }
        return null;
      }
      await waitForDone();
      await fetchCategories();
      state.page = 1;
      loadImages();
    }catch(e){
      console.error('Failed to load annotations', e); alert('Failed to load annotations.json: '+(e && e.message ? e.message : e));
    } finally {
      hideSpinner();
    }
  });
  document.getElementById('closeModal').addEventListener('click', closeModal);
  document.getElementById('closeSkipped').addEventListener('click', hideSkippedModal);
  document.getElementById('modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.getElementById('skippedModal').addEventListener('click', (e) => { if (e.target.id === 'skippedModal') hideSkippedModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  // do not load images initially until a file is loaded
});
