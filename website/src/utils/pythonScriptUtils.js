// loadPyodide.js

// Declare a global variable to hold the Pyodide instance
// This allows other scripts or event listeners in your HTML to access it
window.pyodide = null;
window.isPyodideLoading = false; // Flag to prevent multiple simultaneous loads

// Global function to update a status message on the page
function updateStatus(message) {
    const statusElement = document.getElementById('pyodide-status');
    if (statusElement) {
        statusElement.textContent = message;
    }
}

/**
 * Initializes and loads the Pyodide interpreter.
 * Once loaded, it enables the 'runButton'.
 */
export async function initializePyodideAndSetup() {
    if (window.pyodide) {
        console.warn("Pyodide is already loaded.");
        return window.pyodide;
    }
    if (window.isPyodideLoading) {
        console.warn("Pyodide is already loading. Waiting for it to finish...");
        return new Promise(resolve => {
            const checkInterval = setInterval(() => {
                if (window.pyodide) {
                    clearInterval(checkInterval);
                    resolve(window.pyodide);
                }
            }, 100);
        });
    }

    window.isPyodideLoading = true;
    updateStatus("Loading Pyodide core...");

    try {
        // Dynamically add the Pyodide script if it's not already there
        // This makes sure `loadPyodide` function is available
        if (typeof window.loadPyodide === 'undefined') {
            const script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";
            document.head.appendChild(script);

            await new Promise(resolve => {
                script.onload = resolve;
                script.onerror = (e) => {
                    console.error("Failed to load Pyodide CDN script:", e);
                    throw new Error("Failed to load Pyodide CDN script.");
                };
            });
        }

        window.pyodide = await window.loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/",
            stdout: console.log, // Redirect Python's stdout to console
            stderr: console.error, // Redirect Python's stderr to console
            fullStdLib: true
        });

        updateStatus("Pyodide loaded. Click 'Run Python Script'.");
        const runButton = document.getElementById('runButton');
        if (runButton) {
            runButton.disabled = false; // Enable the button
        }
        console.log("Pyodide instance available.");
        return window.pyodide;

    } catch (error) {
        console.error("Failed to load Pyodide:", error);
        updateStatus(`Error loading Pyodide: ${error.message}`);
        window.pyodide = null; // Ensure pyodide is null on failure
        throw error;
    } finally {
        window.isPyodideLoading = false;
    }
}

/**
 * Fetches and runs a Python script from a given URL using the loaded Pyodide instance.
 * Updates the output and status elements.
 * @param {string} filepath - The URL of the Python file to execute.
 */
export async function runPythonFile(filepath) {
    if (!window.pyodide) {
        updateStatus("Pyodide is not loaded yet! Please wait.");
        return;
    }

    try {
        updateStatus(`Fetching and running ${filepath}...`);

        // Fetch the content of the Python file
        const response = await fetch(filepath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} when fetching ${filepath}`);
        }
        const pythonCode = await response.text();

        // Run the Python code
        // For print() statements, they will go to the browser console.
        // If you need to capture to a specific HTML element, you'll need
        // to redirect sys.stdout within Pyodide as shown in previous examples.
        await window.pyodide.runPythonAsync(pythonCode);

        updateStatus("Python script executed. Check console for direct prints.");

        // Example: Call a function (e.g., 'greet') from the executed script
        // This assumes 'my_script.py' has a 'greet' function defined globally.
        await window.pyodide.runPythonAsync(`
            if 'greet' in globals():
                py_result = greet("Browser User from JS")
            else:
                py_result = "Python function 'greet' not found."
        `);
        const greeting = window.pyodide.globals.get("py_result");

    } catch (error) {
        updateStatus(`Error executing script: ${error.message}`);
        console.error("Error executing Python script:", error);
    }
};