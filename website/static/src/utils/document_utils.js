export function showOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.classList.add('visible');
}


export function hideOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.classList.remove('visible');
}