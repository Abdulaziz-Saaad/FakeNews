// src/components/AddStatement.js
import React, { useState } from 'react';
import axios from 'axios';

const AddStatement = () => {
    const [statement, setStatement] = useState('');
    const [label, setLabel] = useState('1');
    const [message, setMessage] = useState('');
    const [accuracy, setAccuracy] = useState('');
    const [selectedModel, setSelectedModel] = useState('LogisticRegression');
    const [datasetSelectedModel, setDatasetSelectedModel] = useState('dataset3');

    const handleSelect = (event) => {
      setSelectedModel(event.target.value);
    };

    const handleDatasetSelect = (event) => {
      setDatasetSelectedModel(event.target.value);
    };

    const handleAddStatement = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/add_statement', {
                statement,
                label: parseInt(label)
            });
            setMessage(response.data.message);
            setAccuracy(response.data.new_accuracy);
        } catch (error) {
            console.error("Error adding statement:", error);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-semibold mb-4">Add New Statement</h2>

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
            <div className="mb-4">
                <label className="mr-4">Label:</label>
                <select
                    className="p-2 border border-gray-300 rounded"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                >
                    <option value="1">Not Fake</option>
                    <option value="0">Fake</option>
                </select>
            </div>
            <button
                className="bg-green-500 text-white px-4 py-2 rounded"
                onClick={handleAddStatement}
            >
                Add Statement
            </button>
            {message && (
                <div className="mt-4 text-green-600">
                    {message}
                    <p>The new Model Accuracy: <strong>{(accuracy * 100).toFixed(2)}%</strong></p>
                </div>
            )}
        </div>
    );
}

export default AddStatement;
