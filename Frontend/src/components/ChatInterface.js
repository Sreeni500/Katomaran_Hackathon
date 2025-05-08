import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaPaperPlane, FaRobot, FaUser, FaSpinner } from 'react-icons/fa';
import io from 'socket.io-client';

const SOCKET_URL = process.env.REACT_APP_SOCKET_URL || 'http://localhost:3002';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  // Connect to socket and set up event handlers
  useEffect(() => {
    try {
      // Initialize socket connection
      socketRef.current = io(SOCKET_URL);
      
      // Join the chat room
      socketRef.current.emit('join', 'chat');
      
      // Listen for messages
      socketRef.current.on('message', (data) => {
        try {
          const parsedData = JSON.parse(data);
          if (parsedData.response) {
            addMessage('bot', parsedData.response);
            setIsProcessing(false);
          }
          if (parsedData.error) {
            setError(parsedData.error);
            setIsProcessing(false);
          }
        } catch (err) {
          console.error('Error parsing socket data:', err);
          setIsProcessing(false);
        }
      });
      
      // Listen for errors
      socketRef.current.on('error', (data) => {
        try {
          const parsedData = JSON.parse(data);
          setError(parsedData.message || 'Socket error');
          setIsProcessing(false);
        } catch (err) {
          setError('Socket error');
          setIsProcessing(false);
        }
      });
      
    } catch (err) {
      console.error('Socket connection error:', err);
      setError('Failed to connect to server');
    }
    
    // Cleanup function
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);
  
  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Add a message to the chat
  const addMessage = (sender, text) => {
    setMessages(prev => [...prev, { sender, text, timestamp: new Date() }]);
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!inputValue.trim() || isProcessing) return;
    
    // Add user message to chat
    addMessage('user', inputValue);
    
    // Set processing state
    setIsProcessing(true);
    setError(null);
    
    // Send message to server
    if (socketRef.current) {
      socketRef.current.emit('message', JSON.stringify({
        room: 'chat',
        payload: {
          query: inputValue,
          chat_history: messages.map(msg => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text
          }))
        }
      }));
    } else {
      setError('Socket not connected');
      setIsProcessing(false);
    }
    
    // Clear input
    setInputValue('');
  };
  
  // Example queries
  const exampleQueries = [
    "Who was registered today?",
    "When was John recognized last?",
    "How many face recognitions happened yesterday?",
    "List all face registrations this week",
    "What time was Sarah recognized?"
  ];
  
  // Handle example query click
  const handleExampleClick = (query) => {
    setInputValue(query);
  };
  
  return (
    <div>
      <h2 className="text-xl md:text-2xl font-semibold text-gray-800 mb-4">AI Chat Interface</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Chat history */}
        <div className="md:col-span-2 bg-white rounded-lg shadow overflow-hidden flex flex-col h-[500px]">
          {/* Messages area */}
          <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-gray-400">
                <FaRobot className="text-5xl mb-3 text-primary-300" />
                <p className="text-center">No messages yet. Ask a question about face recognition activity.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`flex items-start max-w-[80%] ${message.sender === 'user' ? 'chat-message-user' : 'chat-message-bot'}`}>
                      {message.sender === 'bot' && <FaRobot className="mt-1 mr-2 text-gray-500 flex-shrink-0" />}
                      <div>
                        <p className="whitespace-pre-wrap">{message.text}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          {new Date(message.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                      {message.sender === 'user' && <FaUser className="mt-1 ml-2 text-primary-500 flex-shrink-0" />}
                    </div>
                  </motion.div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          
          {/* Input area */}
          <div className="p-3 border-t border-gray-200 bg-white">
            <form onSubmit={handleSubmit} className="flex space-x-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={isProcessing}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                placeholder="Ask about face recognition activity..."
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isProcessing}
                className={`px-4 py-2 rounded-md flex items-center justify-center ${
                  !inputValue.trim() || isProcessing
                    ? 'bg-gray-300 cursor-not-allowed'
                    : 'bg-primary-600 text-white hover:bg-primary-700'
                } transition-colors`}
              >
                {isProcessing ? (
                  <FaSpinner className="animate-spin" />
                ) : (
                  <FaPaperPlane />
                )}
              </button>
            </form>
            
            {error && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-2 p-2 rounded-md bg-red-100 text-red-800 text-sm"
              >
                <p>{error}</p>
              </motion.div>
            )}
          </div>
        </div>
        
        {/* Information sidebar */}
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
          <h3 className="text-lg font-medium text-gray-800 mb-2">About the Chat</h3>
          <p className="text-gray-600 mb-4">
            This AI assistant can answer questions about face registration and 
            recognition events based on the system's logs.
          </p>
          
          <h4 className="font-medium text-gray-700 mb-2">Example Questions</h4>
          <div className="space-y-2">
            {exampleQueries.map((query, index) => (
              <div 
                key={index}
                onClick={() => handleExampleClick(query)}
                className="p-2 bg-white rounded border border-gray-200 text-sm text-gray-700 cursor-pointer hover:bg-primary-50 hover:border-primary-200 transition-colors"
              >
                {query}
              </div>
            ))}
          </div>
          
          <div className="mt-6">
            <h4 className="font-medium text-gray-700 mb-2">Capabilities</h4>
            <ul className="space-y-1 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                Query face registration history
              </li>
              <li className="flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                Ask about recent recognitions
              </li>
              <li className="flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                Get information about specific people
              </li>
              <li className="flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                Ask about recognition timestamps
              </li>
              <li className="flex items-start">
                <span className="text-primary-600 mr-2">•</span>
                Learn about face recognition patterns
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 