document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const resultDiv = document.getElementById('result');
    const priceValue = document.getElementById('price-value');
    const confidenceValue = document.getElementById('confidence-value');
    
    // Loading state
    submitBtn.textContent = 'Calculating...';
    submitBtn.disabled = true;
    resultDiv.classList.add('hidden');
    
    // Gather data
    const formData = new FormData(form);
    const amenities = formData.getAll('amenities');
    
    const data = {
        area: parseFloat(formData.get('area')),
        location: formData.get('location'),
        bedrooms: parseInt(formData.get('bedrooms')),
        bathrooms: parseFloat(formData.get('bathrooms')),
        amenities: amenities
    };
    
    try {
        const response = await fetch('http://localhost:8000/predict-price', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display result
        priceValue.textContent = result.predicted_price.toLocaleString();
        confidenceValue.textContent = result.confidence_range;
        resultDiv.classList.remove('hidden');
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please try again.');
    } finally {
        submitBtn.textContent = 'Predict Price';
        submitBtn.disabled = false;
    }
});
