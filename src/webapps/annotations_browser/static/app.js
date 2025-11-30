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

function showSpinner(msg) {
  const gp = document.getElementById('glasspane');
  if (!gp) return;
  gp.classList.remove('hidden');
  gp.setAttribute('aria-hidden', 'false');
}

function hideSpinner() {
  const gp = document.getElementById('glasspane');
  if (!gp) return;
  gp.classList.add('hidden');
  gp.setAttribute('aria-hidden', 'true');
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
  document.getElementById('loadFileBtn').addEventListener('click', async () => {
    const sel = document.getElementById('fileSelect');
    const fp = sel.value;
    if (!fp) return alert('Please select an annotations.json file');
    try{
      showSpinner();
      const r = await fetch('/api/load_annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_path: fp})});
      if(!r.ok) throw new Error('Load failed');
      const j = await r.json();
      // after load, fetch categories and images
      await fetchCategories();
      state.page = 1;
      loadImages();
    }catch(e){
      console.error('Failed to load annotations', e); alert('Failed to load annotations.json');
    } finally {
      hideSpinner();
    }
  });
  document.getElementById('closeModal').addEventListener('click', closeModal);
  document.getElementById('modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  // do not load images initially until a file is loaded
});
