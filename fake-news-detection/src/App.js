import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Predict from './components/Predict';
import AddStatement from './components/AddStatement';

const App = () => {
    return (
        <Router>
            <div className="bg-gray-100 min-h-screen flex flex-col">
                <nav className="bg-blue-500 p-4 text-white">
                    <div className="container mx-auto flex justify-between">
                        <Link to="/" className="font-semibold text-xl">Fake News Detection</Link>
                        <div>
                            <Link to="/" className="mx-2">Predict</Link>
                            <Link to="/add" className="mx-2">Add Statement</Link>
                        </div>
                    </div>
                </nav>
                <div className="container mx-auto p-4 flex-grow">
                    <Routes>
                        <Route path="/" element={<Predict />} />
                        <Route path="/add" element={<AddStatement />} />
                    </Routes>
                </div>
            </div>
        </Router>
    );
}

export default App;