const nav=document.getElementById('nav');
const updateNav=()=>nav.classList.toggle('scrolled',window.scrollY>window.innerHeight-90);
window.addEventListener('scroll',updateNav,{passive:true});updateNav();
const video=document.getElementById('hero-video'),toggle=document.getElementById('play-toggle');
function sync(){toggle.textContent=video.paused?'Play':'Pause';toggle.setAttribute('aria-label',video.paused?'Play background video':'Pause background video');}
video.addEventListener('play',sync);video.addEventListener('pause',sync);
toggle.addEventListener('click',()=>{if(video.paused)video.play().catch(sync);else video.pause();});
if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)video.pause();sync();
