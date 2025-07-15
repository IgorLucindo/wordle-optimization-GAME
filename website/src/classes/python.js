export class Python {
    constructor() {
        this.pyodide = null;
        this.isPyodideLoading = false;
    }


    // Initializes and loads the Pyodide interpreter and required packages.
    async init() {
        if (this.pyodide || this.isPyodideLoading) {
            console.warn("Pyodide is already loaded or loading.");
            return;
        }

        this.isPyodideLoading = true;

        try {
            // Load the Pyodide main script
            if (typeof window.loadPyodide === 'undefined') {
                const script = document.createElement('script');
                script.src = "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js";
                document.head.appendChild(script);
                await new Promise((resolve, reject) => {
                    script.onload = resolve;
                    script.onerror = (e) => reject(new Error("Failed to load Pyodide CDN script."));
                });
            }

            this.pyodide = await window.loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/",
                stdout: console.log, // Redirect Python's stdout to the browser console
                stderr: console.error, // Redirect Python's stderr to the browser console
            });

            await this.pyodide.loadPackage("scipy");

            const runButton = document.getElementById('runButton');
            if (runButton) runButton.disabled = false;

            return this.pyodide;

        } catch (error) {
            console.error("Failed to load Pyodide:", error);
            this.pyodide = null;
            throw error;
        } finally {
            this.isPyodideLoading = false;
        }
    }


    // Run python file
    async run(filepath) {
        if (!this.pyodide) {
            // Try to initialize it now
            await this.init();
            if (!this.pyodide) return; // Exit if initialization failed
        }

        try {
            const response = await fetch(filepath);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} when fetching ${filepath}`);
            }
            const pythonCode = await response.text();

            // Run the Python script. It will run an async function and set a global variable.
            await this.pyodide.runPythonAsync(pythonCode);

            // --- GETTING RESULTS BACK FROM PYTHON ---
            // The python script (main.py) will set a global variable named 'pyodide_results'.
            const results = this.pyodide.globals.get('pyodide_results');

            if (results) {
                // Convert the PyProxy map to a JavaScript object to easily access the data
                const jsResults = results.toJs({ dict_converter: Object.fromEntries });
                console.log("Results retrieved from Python:", jsResults);
                results.destroy(); // Clean up the PyProxy to free memory
            }

        } catch (error) {
            console.error("Error executing Python script:", error);
        }
    }
}