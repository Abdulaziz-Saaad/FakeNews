// src/components/AddStatement.js
import React, { useState } from 'react';
import axios from 'axios';

const AddStatement = () => {
    const [statement, setStatement] = useState('');
    const [label, setLabel] = useState('1');
    const [message, setMessage] = useState('');
    const [accuracy, setAccuracy] = useState('');

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
