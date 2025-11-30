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
    el.appendChild(img);
    const fname = document.createElement('div');
    fname.className = 'fname';
    fname.textContent = item.file_name;
    el.appendChild(fname);
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
      const r = await fetch('/api/load_annotations', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_path: fp})});
      if(!r.ok) throw new Error('Load failed');
      const j = await r.json();
      // after load, fetch categories and images
      await fetchCategories();
      state.page = 1;
      loadImages();
    }catch(e){
      console.error('Failed to load annotations', e); alert('Failed to load annotations.json');
    }
  });
  document.getElementById('closeModal').addEventListener('click', closeModal);
  document.getElementById('modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

  // do not load images initially until a file is loaded
});
