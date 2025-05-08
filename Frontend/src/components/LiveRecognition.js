import React, { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { FaVideo, FaVideoSlash, FaCog, FaExclamationTriangle, FaUser, FaUserCheck } from 'react-icons/fa';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3002/api';

const LiveRecognition = () => {
  const webcamRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [recognizedFaces, setRecognizedFaces] = useState([]);
  const [isPaused, setIsPaused] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [error, setError] = useState(null);
  const [quality, setQuality] = useState("high"); // low, normal, high
  const [debugInfo, setDebugInfo] = useState(null);
  const [capturedFrame, setCapturedFrame] = useState(null);
  const [recognitionFrequency, setRecognitionFrequency] = useState(500); // ms between recognitions
  const [loading, setLoading] = useState(false);
  
  const recognitionIntervalRef = useRef(null);
  
  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };
  
  const performRecognition = useCallback(async () => {
    if (!webcamRef.current || !isActive || isPaused || loading) return;
    
    try {
      setLoading(true);
      setDebugInfo("Capturing frame for recognition...");
      
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        setDebugInfo("Failed to capture screenshot");
        return;
      }
      
      setCapturedFrame(imageSrc);
      setDebugInfo("Sending frame to recognition API...");
      
      const response = await axios.post(`${API_URL}/recognize-face`, {
        frame_data: imageSrc
      });
      
      if (response.data.error) {
        setError(response.data.error);
        setDebugInfo(`Recognition error: ${response.data.error}`);
      } else {
        setError(null);
        setRecognizedFaces(response.data.faces || []);
        setDebugInfo(`Recognition complete: ${response.data.faces?.length || 0} faces detected`);
      }
    } catch (error) {
      setError(`Connection error: ${error.message}`);
      setDebugInfo(`API error: ${JSON.stringify(error.response?.data || error.message)}`);
      console.error("Error recognizing face:", error);
    } finally {
      setLoading(false);
    }
  }, [isActive, isPaused, loading]);
  
  useEffect(() => {
    if (isActive && !isPaused) {
      recognitionIntervalRef.current = setInterval(() => {
        performRecognition();
      }, recognitionFrequency);
      
      // Initial recognition
      performRecognition();
    }
    
    return () => {
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, [isActive, isPaused, performRecognition, recognitionFrequency]);
  
  const toggleRecognition = () => {
    setIsActive(prev => !prev);
  };
  
  const togglePause = () => {
    setIsPaused(prev => !prev);
  };
  
  const toggleFullScreen = () => {
    setIsFullScreen(prev => !prev);
  };
  
  return (
    <div className={`${isFullScreen ? 'fixed inset-0 z-50 bg-black p-4' : ''}`}>
      <div className={`${isFullScreen ? 'h-full flex flex-col' : ''}`}>
        <div className="flex justify-between items-center mb-4">
          <h2 className={`text-xl md:text-2xl font-semibold ${isFullScreen ? 'text-white' : 'text-gray-800'}`}>
            Live Face Recognition
          </h2>
          <div className="flex space-x-2">
            <button
              onClick={toggleRecognition}
              className={`px-3 py-1 rounded-md ${isActive ? 'bg-red-500 hover:bg-red-600' : 'bg-primary-500 hover:bg-primary-600'} text-white flex items-center`}
            >
              {isActive ? <><FaVideoSlash className="mr-1" /> Stop</> : <><FaVideo className="mr-1" /> Start</>}
            </button>
            {isActive && (
              <>
                <button
                  onClick={togglePause}
                  className={`px-3 py-1 rounded-md ${isPaused ? 'bg-green-500 hover:bg-green-600' : 'bg-yellow-500 hover:bg-yellow-600'} text-white`}
                >
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={toggleFullScreen}
                  className="px-3 py-1 rounded-md bg-gray-700 hover:bg-gray-800 text-white"
                >
                  {isFullScreen ? 'Exit Full' : 'Full Screen'}
                </button>
              </>
            )}
          </div>
        </div>
        
        <div className={`relative webcam-container bg-gray-100 rounded-lg overflow-hidden shadow-lg ${isFullScreen ? 'flex-grow' : ''}`}>
          {/* Main webcam feed */}
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className={`${isFullScreen ? 'w-full h-full object-cover' : 'w-full'}`}
            screenshotQuality={quality === "high" ? 1.0 : quality === "normal" ? 0.9 : 0.8}
            forceScreenshotSourceSize={true}
          />
          
          {/* Recognition overlay for face outlines */}
          <div className="absolute inset-0 pointer-events-none">
            {recognizedFaces.map((face, index) => {
              if (!face.facial_area) return null;
              
              const { x, y, w, h } = face.facial_area;
              
              return (
                <div
                  key={index}
                  className={`absolute border-2 ${face.registered ? 'border-green-500' : 'border-yellow-500'}`}
                  style={{
                    left: `${x}px`,
                    top: `${y}px`,
                    width: `${w}px`,
                    height: `${h}px`,
                    transform: 'scale(calc(100vw / 1280))', // Scale based on screen size
                    transformOrigin: 'top left'
                  }}
                >
                  <div 
                    className={`absolute -top-7 left-0 text-xs px-2 py-1 rounded-t-md ${
                      face.registered ? 'bg-green-500' : 'bg-yellow-500'
                    } text-white whitespace-nowrap flex items-center`}
                  >
                    {face.registered ? (
                      <><FaUserCheck className="mr-1" /> {face.label} ({Math.round(face.confidence * 100)}%)</>
                    ) : (
                      <><FaUser className="mr-1" /> Unknown</>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Status overlay */}
          <div className="absolute top-2 right-2">
            <div className={`px-2 py-1 rounded-md text-xs ${isActive ? (isPaused ? 'bg-yellow-500' : 'bg-green-500') : 'bg-gray-500'} text-white`}>
              {isActive ? (isPaused ? 'Paused' : 'Active') : 'Inactive'}
            </div>
          </div>
          
          {/* Loading indicator */}
          {loading && (
            <div className="absolute inset-0 bg-black bg-opacity-10 flex items-center justify-center">
              <div className="bg-white bg-opacity-80 rounded-lg p-2 text-xs text-gray-700">
                Processing...
              </div>
            </div>
          )}
        </div>
        
        {/* Controls and settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <div className="space-y-4">
            {error && (
              <div className="p-2 rounded-md bg-red-50 border border-red-200 text-red-800 text-sm flex items-center">
                <FaExclamationTriangle className="mr-2 text-red-500" />
                <p>{error}</p>
              </div>
            )}
            
            {debugInfo && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-2 rounded-md bg-gray-50 border border-gray-200 text-gray-600 text-xs"
              >
                <p className="font-mono overflow-x-auto">{debugInfo}</p>
              </motion.div>
            )}
            
            {/* Fast capture preview */}
            {capturedFrame && (
              <div className="w-32 h-24 bg-gray-100 rounded border border-gray-300 overflow-hidden">
                <img 
                  src={capturedFrame} 
                  alt="Last processed frame" 
                  className="w-full h-full object-cover"
                />
              </div>
            )}
            
            {/* Settings panel */}
            <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
              <div className="flex items-center mb-2">
                <FaCog className="text-gray-500 mr-2" />
                <h3 className="text-sm font-medium text-gray-700">Recognition Settings</h3>
              </div>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Recognition Speed</label>
                  <select 
                    value={recognitionFrequency}
                    onChange={(e) => setRecognitionFrequency(Number(e.target.value))}
                    className="w-full text-sm p-1 border border-gray-300 rounded"
                  >
                    <option value={200}>Very Fast (200ms)</option>
                    <option value={500}>Fast (500ms)</option>
                    <option value={1000}>Normal (1s)</option>
                    <option value={2000}>Slow (2s)</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Image Quality</label>
                  <div className="flex justify-between text-sm">
                    <label className="flex items-center text-gray-600 cursor-pointer">
                      <input
                        type="radio"
                        name="quality"
                        checked={quality === "low"}
                        onChange={() => setQuality("low")}
                        className="mr-1"
                      />
                      Low
                    </label>
                    <label className="flex items-center text-gray-600 cursor-pointer">
                      <input
                        type="radio"
                        name="quality"
                        checked={quality === "normal"}
                        onChange={() => setQuality("normal")}
                        className="mr-1"
                      />
                      Normal
                    </label>
                    <label className="flex items-center text-gray-600 cursor-pointer">
                      <input
                        type="radio"
                        name="quality"
                        checked={quality === "high"}
                        onChange={() => setQuality("high")}
                        className="mr-1"
                      />
                      High
                    </label>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <h3 className="text-lg font-medium text-gray-800 mb-2">Recognition Results</h3>
              
              {recognizedFaces.length > 0 ? (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {recognizedFaces.map((face, index) => (
                    <motion.div 
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`p-2 rounded-md border ${
                        face.registered ? 'bg-green-50 border-green-200 text-green-800' : 'bg-yellow-50 border-yellow-200 text-yellow-800'
                      } text-sm flex items-center`}
                    >
                      {face.registered ? (
                        <>
                          <FaUserCheck className="mr-2 text-green-500" />
                          <div>
                            <p className="font-medium">{face.label}</p>
                            <p className="text-xs">Confidence: {Math.round(face.confidence * 100)}%</p>
                          </div>
                        </>
                      ) : (
                        <>
                          <FaUser className="mr-2 text-yellow-500" />
                          <div>
                            <p className="font-medium">Unknown Person</p>
                            <p className="text-xs">Not registered in the system</p>
                          </div>
                        </>
                      )}
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-gray-500 italic">
                  {isActive ? (isPaused ? 'Recognition paused' : 'No faces detected') : 'Start recognition to detect faces'}
                </div>
              )}
            </div>
            
            <div className="bg-blue-50 p-3 border border-blue-100 rounded-md">
              <h4 className="text-sm font-medium text-blue-800 mb-1">Tips:</h4>
              <ul className="text-xs text-blue-700 space-y-1 pl-4 list-disc">
                <li>Face the camera directly for better recognition</li>
                <li>Ensure good lighting on your face</li>
                <li>Slow down recognition if performance is sluggish</li>
                <li>Make sure the person is registered before recognition</li>
                <li>Higher quality means better recognition but slower performance</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveRecognition; 