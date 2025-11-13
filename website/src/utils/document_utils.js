export function showOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.classList.add('visible');
}


export function hideOverlay() {
    const overlay = document.getElementById('overlay');
    overlay.classList.remove('visible');
}


let tooltipTimeout;
export function showTooltip(btn, msg) {
    const tooltip = document.getElementById('tooltipText');

    // Clear timeout
    if (tooltipTimeout) {
        clearTimeout(tooltipTimeout);
    }

    // Change message
    tooltip.innerHTML = msg;

    // Show the tooltip
    tooltip.classList.add('visible');

    // Position the tooltip
    const rect = btn.getBoundingClientRect(); 
    const tooltipWidth = tooltip.offsetWidth;
    const tooltipTop = rect.bottom + window.scrollY + 18; 
    const tooltipLeft = rect.left + (rect.width / 2) - (tooltipWidth / 2) + window.scrollX;
    tooltip.style.top = `${tooltipTop}px`;
    tooltip.style.left = `${tooltipLeft}px`;

    // Hide after fixed time
    tooltipTimeout = setTimeout(() => {
        tooltip.classList.remove('visible');
        tooltipTimeout = null;
    }, 3000);
}