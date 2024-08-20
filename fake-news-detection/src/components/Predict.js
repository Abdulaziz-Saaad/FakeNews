// src/components/Predict.js
import React, { useState } from 'react';
import axios from 'axios';

const Predict = () => {
    const [statement, setStatement] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [accuracy, setAccuracy] = useState(null);
    const [selectedModel, setSelectedModel] = useState('LogisticRegression');
    const [datasetSelectedModel, setDatasetSelectedModel] = useState('dataset3');

    const handleSelect = (event) => {
      setSelectedModel(event.target.value);
    };

    const handleDatasetSelect = (event) => {
      setDatasetSelectedModel(event.target.value);
    };

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

            <div className="flex justify-center">
              <select
                id="model-select"
                value={selectedModel}
                onChange={handleSelect}
                className="mt-1 flex m-3 py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="">Choose a model</option>
                <option value="LogisticRegression">Logistic Regression</option>
                <option value="RandomForestClassifier">RandomForestClassifier</option>
                <option value="LinearSVC">LinearSVC</option>
                <option value="DecisionTreeClassifier">DecisionTreeClassifier</option>
              </select>

              <select
                id="model-select"
                value={datasetSelectedModel}
                onChange={handleDatasetSelect}
                className="mt-1 flex my-3 py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              >
                <option value="">Choose a dataset</option>
                <option value="dataset1">Dataset 1</option>
                <option value="dataset2">Dataset 2</option>
                <option value="dataset3">Dataset 3</option>
              </select>
            </div>
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
