export async function callPythonScript(route, inputData={}) {
    try {
        const response = await fetch(route, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            // If the server response was not ok (e.g., 404, 500)
            const errorData = await response.json(); // Assuming server sends JSON error
            throw new Error(`Server error: ${response.status} - ${errorData.message || response.statusText}`);
        }

        return await response.json();
    }
    catch (error) {
        console.error('Error calling Python script:', error);
        throw error;
    }
}