const body = document.body;
const toggleBtn = document.getElementById("theme-toggle");

// Function to update icon
function updateIcon() {
  toggleBtn.innerHTML = body.classList.contains("dark-theme")
    ? '<i class="fas fa-sun"></i>'
    : '<i class="fas fa-moon"></i>';
}

// --- INITIAL THEME CHECK (Your existing code) ---
const savedTheme = localStorage.getItem("theme");
if (savedTheme) {
  if (savedTheme === "dark") body.classList.add("dark-theme");
} else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
  body.classList.add("dark-theme");
}

updateIcon();

// --- NEW CODE: ENABLE TRANSITIONS AFTER INITIAL LOAD ---
// This ensures the first theme change has no animation.
setTimeout(() => {
  body.classList.add("enable-transition");
}, 10); // A tiny delay is enough

// Toggle theme on click
toggleBtn.addEventListener("click", () => {
  body.classList.toggle("dark-theme");
  localStorage.setItem(
    "theme",
    body.classList.contains("dark-theme") ? "dark" : "light"
  );
  updateIcon();
});