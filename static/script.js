function toggleChat() {
    const cb = document.getElementById("chatbox");
    cb.classList.toggle("hidden");
    const hidden = cb.classList.contains('hidden');
    cb.setAttribute('aria-hidden', hidden ? 'true' : 'false');
}

function fillInput(text) {
    document.getElementById("inputText").value = text;
}
// Close mobile nav when a link is clicked
document.addEventListener('click', (e)=>{
  if(e.target.matches('.nav-item')) document.body.classList.remove('nav-open');
});

// Show loading spinner when predict form is submitted (prevents multiple submits)
document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('predictForm');
  if(!form) return;
  const btn = document.getElementById('predictBtn');
  const spinner = document.getElementById('btnSpinner');
  const label = document.getElementById('btnLabel');

  form.addEventListener('submit', ()=>{
    if(btn) btn.disabled = true;
    if(spinner) spinner.style.display = 'inline-block';
    if(label) label.textContent = 'Predicting...';
  });
});