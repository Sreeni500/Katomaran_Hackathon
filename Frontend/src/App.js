import React, { useState } from 'react';
import { motion } from 'framer-motion';
import FaceRegistration from './components/FaceRegistration';
import LiveRecognition from './components/LiveRecognition';
import ChatInterface from './components/ChatInterface';

const tabs = [
  { id: 'register', label: 'Face Registration' },
  { id: 'recognize', label: 'Live Recognition' },
  { id: 'chat', label: 'AI Chat' },
];

function App() {
  const [activeTab, setActiveTab] = useState('register');

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto py-4 px-4 md:px-6">
          <h1 className="text-2xl md:text-3xl font-bold text-gray-900">
            <span className="text-primary-600">Katomaran</span> Face Recognition Platform
          </h1>
        </div>
      </header>

      <div className="container mx-auto py-6 px-4 md:px-6">
        {/* Tab Navigation */}
        <div className="flex space-x-1 border-b border-gray-200 mb-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm md:text-base font-medium transition-colors duration-200
                ${activeTab === tab.id ? 'tab-active' : 'text-gray-500 hover:text-gray-700'}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content with Framer Motion Animations */}
        <div className="relative overflow-hidden bg-white rounded-lg shadow-lg p-4 md:p-6">
          {activeTab === 'register' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <FaceRegistration />
            </motion.div>
          )}

          {activeTab === 'recognize' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <LiveRecognition />
            </motion.div>
          )}

          {activeTab === 'chat' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              <ChatInterface />
            </motion.div>
          )}
        </div>
      </div>

      <footer className="bg-white border-t border-gray-200 mt-8">
        <div className="container mx-auto py-4 px-4 md:px-6 text-center text-gray-500 text-sm">
          <p>This project is a part of a hackathon run by <a href="https://katomaran.com" className="text-primary-600 hover:underline">https://katomaran.com</a></p>
        </div>
      </footer>
    </div>
  );
}

export default App; 