// src/components/Predict.js
import React, { useState } from 'react';
import axios from 'axios';

const Predict = () => {
    const [statement, setStatement] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [accuracy, setAccuracy] = useState(null);

    const handlePredict = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', { statement });
            setPrediction(response.data.prediction);
            setAccuracy(response.data.accuracy);
        } catch (error) {
            console.error("Error making prediction:", error);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Predict Fake News</h2>
            <textarea
                className="w-full p-2 border border-gray-300 rounded mb-4"
                rows="4"
                placeholder="Enter news statement..."
                value={statement}
                onChange={(e) => setStatement(e.target.value)}
            />
            <button
                className="bg-blue-500 text-white px-4 py-2 rounded"
                onClick={handlePredict}
            >
                Predict
            </button>
            
            {prediction && (
                <div className="mt-4">
                    <p>Prediction: <strong>{prediction == "True" ? 'Not Fake' : 'Fake'}</strong></p>
                    <p>Model Accuracy: <strong>{(accuracy * 100).toFixed(2)}%</strong></p>
                </div>
            )}
        </div>
    );
}

export default Predict;
