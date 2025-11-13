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
    const viewportWidth = window.innerWidth;
    const leftBoundary = 10;
    const rightBoundary = viewportWidth - tooltipWidth - 10;
    const tooltipTop = rect.bottom + window.scrollY + 18;
    let tooltipLeft = rect.left + (rect.width / 2) - (tooltipWidth / 2) + window.scrollX;
    if (tooltipLeft < leftBoundary) tooltipLeft = leftBoundary;
    if (tooltipLeft > rightBoundary) tooltipLeft = rightBoundary;
    tooltip.style.top = `${tooltipTop}px`;
    tooltip.style.left = `${tooltipLeft}px`;

    // Position before
    const btnCenter = rect.left + (rect.width / 2) + window.scrollX;
    const arrowPositionInTooltip = btnCenter - tooltipLeft;
    const arrowLeftPercentage = (arrowPositionInTooltip / tooltipWidth) * 100;
    tooltip.style.setProperty('--arrow-left-position', `${arrowLeftPercentage}%`);

    // Hide after fixed time
    tooltipTimeout = setTimeout(() => {
        tooltip.classList.remove('visible');
        tooltipTimeout = null;
    }, 3000);
}