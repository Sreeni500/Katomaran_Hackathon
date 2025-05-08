import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { FaCamera, FaSpinner, FaCheck, FaExclamationTriangle, FaUserPlus, FaRedo } from 'react-icons/fa';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3002/api';

const FaceRegistration = () => {
  const [name, setName] = useState('');
  const [capturedImage, setCapturedImage] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [registrationStatus, setRegistrationStatus] = useState(null);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [detectionError, setDetectionError] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [quality, setQuality] = useState("high"); // low, normal, high - default to high
  const [debugInfo, setDebugInfo] = useState(null);
  const webcamRef = useRef(null);
  const faceCheckTimerRef = useRef(null);
  
  const videoConstraints = {
    width: 640, // Reduced from 1280 for better performance
    height: 480, // Reduced from 720 for better performance
    facingMode: "user",
    frameRate: 30
  };
  
  // Periodically check for faces in preview
  const checkForFaces = useCallback(async () => {
    if (isCapturing || !webcamRef.current) return;
    
    try {
      const imageSrc = webcamRef.current.getScreenshot({
        width: 640,
        height: 480,
        quality: quality === "high" ? 0.9 : quality === "normal" ? 0.8 : 0.7
      });
      
      if (!imageSrc) {
        setDebugInfo("Failed to capture webcam frame");
        return;
      }
      
      setDebugInfo("Checking for faces...");
      const response = await axios.post(`${API_URL}/recognize-face`, {
        frame_data: imageSrc,
        check_only: true // New flag to indicate we only want face detection
      }, {
        timeout: 5000 // 5 second timeout
      });
      
      if (response.data.success === false) {
        setIsFaceDetected(false);
        setDetectionError(response.data.message || "Failed to detect face. Please adjust position or lighting.");
        setDebugInfo(`Detection error: ${response.data.message || "Unknown error"}`);
      } 
      else if (response.data.faces && response.data.faces.length > 0) {
        setIsFaceDetected(true);
        setDetectionError(null);
        setDebugInfo(`Face detected with confidence: ${response.data.faces[0].confidence || 'N/A'}`);
      } else {
        setIsFaceDetected(false);
        setDetectionError("No face detected. Please center your face in the frame and ensure good lighting.");
        setDebugInfo("No face detected in frame");
      }
    } catch (error) {
      console.error('Error detecting face:', error);
      setIsFaceDetected(false);
      setDetectionError(`Error detecting face: ${error.response?.data?.message || error.message || 'Unknown error'}`);
      setDebugInfo(`Detection error: ${JSON.stringify(error.response?.data || error.message)}`);
    }
  }, [isCapturing, quality]);
  
  // Set up interval for face detection in preview
  useEffect(() => {
    faceCheckTimerRef.current = setInterval(checkForFaces, 1000);
    return () => {
      if (faceCheckTimerRef.current) {
        clearInterval(faceCheckTimerRef.current);
      }
    };
  }, [checkForFaces]);
  
  // Countdown effect
  useEffect(() => {
    if (countdown === null) return;
    
    if (countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else {
      captureNow();
    }
  }, [countdown]);
  
  const startCountdown = useCallback(() => {
    if (!name.trim()) {
      setRegistrationStatus({
        success: false,
        message: 'Please enter a name first'
      });
      return;
    }
    
    if (!isFaceDetected && !force_register) {
      setRegistrationStatus({
        success: false,
        message: 'No face detected. Please position your face in the frame.'
      });
      return;
    }
    
    setCountdown(3);
  }, [name, isFaceDetected]);
  
  const captureNow = useCallback(() => {
    setIsCapturing(true);
    setRegistrationStatus(null);
    setCountdown(null);
    
    const imageSrc = webcamRef.current.getScreenshot({
      width: 640,
      height: 480,
      quality: quality === "high" ? 0.9 : quality === "normal" ? 0.8 : 0.7
    });
    
    setCapturedImage(imageSrc);
    registerFace(imageSrc);
  }, [quality]);
  
  const registerFace = async (imageSrc, forceRegister = false) => {
    try {
      setDebugInfo("Registering face...");
      
      // Make sure image has proper format
      let formattedImage = imageSrc;
      if (!formattedImage.startsWith('data:image/jpeg;base64,')) {
        formattedImage = `data:image/jpeg;base64,${imageSrc.split(',').pop() || imageSrc}`;
      }
      
      setDebugInfo(`Sending registration request to ${API_URL}/register-face`);
      
      const response = await axios.post(`${API_URL}/register-face`, {
        name: name,
        frame_data: formattedImage,
        force_register: forceRegister
      }, {
        timeout: 10000 // 10 second timeout for registration
      });
      
      setRegistrationStatus({
        success: response.data.success,
        message: response.data.message || (response.data.success ? "Registration successful!" : "Registration failed.")
      });
      
      setDebugInfo(`Registration response: ${JSON.stringify(response.data)}`);
    } catch (error) {
      console.error('Error registering face:', error);
      const errorMessage = error.response?.data?.message || error.message || 'Error registering face';
      const networkError = error.message.includes('Network Error') ? 
        ' (Connection problem: Make sure the backend servers are running)' : '';
      
      setRegistrationStatus({
        success: false,
        message: errorMessage + networkError
      });
      
      setDebugInfo(`Registration error: ${JSON.stringify(error.response?.data || error.message)}`);
    } finally {
      setIsCapturing(false);
    }
  };
  
  const resetCapture = () => {
    setCapturedImage(null);
    setRegistrationStatus(null);
  };

  return (
    <div>
      <h2 className="text-xl md:text-2xl font-semibold text-gray-800 mb-4">Face Registration</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="mb-4">
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <div className="relative">
              <input
                type="text"
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full pl-3 pr-10 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
                placeholder="Enter person's name"
              />
              <FaUserPlus className="absolute right-3 top-2.5 text-gray-400" />
            </div>
          </div>
          
          <div className="relative webcam-container bg-gray-100 rounded-lg overflow-hidden shadow-lg">
            {!capturedImage ? (
              <>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="w-full rounded-lg"
                  screenshotQuality={quality === "high" ? 1.0 : quality === "normal" ? 0.9 : 0.8}
                  forceScreenshotSourceSize={true}
                />
                
                {/* Face detection indicator */}
                <AnimatePresence>
                  {isFaceDetected ? (
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-2 left-2 bg-green-500 text-white px-2 py-1 rounded-md text-xs flex items-center"
                    >
                      <FaCheck className="mr-1" /> Face detected
                    </motion.div>
                  ) : (
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="absolute top-2 left-2 bg-yellow-500 text-white px-2 py-1 rounded-md text-xs flex items-center"
                    >
                      <FaExclamationTriangle className="mr-1" /> No face detected
                    </motion.div>
                  )}
                </AnimatePresence>
                
                {/* Face detection guide overlay */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="w-[60%] h-[70%] mx-auto mt-[15%] border-2 border-dashed border-primary-300 rounded-full opacity-50"></div>
                </div>
                
                {/* Countdown overlay */}
                <AnimatePresence>
                  {countdown !== null && (
                    <motion.div 
                      initial={{ opacity: 0, scale: 1.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.5 }}
                      className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50"
                    >
                      <motion.div 
                        initial={{ scale: 1.5 }}
                        animate={{ scale: 1 }}
                        transition={{ duration: 0.8 }}
                        className="text-6xl font-bold text-white"
                      >
                        {countdown}
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            ) : (
              <motion.img 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                src={capturedImage} 
                alt="Captured" 
                className="w-full rounded-lg"
              />
            )}
          </div>
          
          <div className="flex flex-col space-y-2">
            {!capturedImage ? (
              <>
                <div className="flex space-x-2">
                  <button
                    onClick={startCountdown}
                    disabled={isCapturing || !name.trim()}
                    className={`flex-1 flex items-center justify-center px-4 py-2 rounded-md ${
                      !name.trim() ? 'bg-gray-300 cursor-not-allowed' : 'bg-primary-600 text-white hover:bg-primary-700'
                    } transition-colors`}
                  >
                    {isCapturing ? (
                      <>
                        <FaSpinner className="animate-spin mr-2" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <FaCamera className="mr-2" />
                        Capture with Countdown
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => capture(false)}
                    disabled={isCapturing || !name.trim()}
                    className={`flex items-center justify-center px-4 py-2 rounded-md ${
                      !name.trim() ? 'bg-gray-300 cursor-not-allowed' : 'bg-secondary-600 text-white hover:bg-secondary-700'
                    } transition-colors`}
                  >
                    <FaCamera className="mr-0" />
                  </button>
                </div>
                
                {/* Force register button - always enabled when name is provided */}
                <button
                  onClick={() => capture(true)}
                  disabled={isCapturing || !name.trim()}
                  className="flex items-center justify-center px-4 py-2 w-full rounded-md bg-red-600 text-white hover:bg-red-700 font-bold transition-colors"
                  style={{padding: '12px', fontSize: '16px'}}
                >
                  <FaUserPlus className="mr-2" size={20} />
                  FORCE REGISTER (Click if normal registration fails)
                </button>
                
                <div className="flex justify-between text-sm">
                  <label className="flex items-center text-gray-600 cursor-pointer">
                    <input
                      type="radio"
                      name="quality"
                      checked={quality === "low"}
                      onChange={() => setQuality("low")}
                      className="mr-1"
                    />
                    Low Quality
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
                    High Quality
                  </label>
                </div>
              </>
            ) : (
              <button
                onClick={resetCapture}
                className="flex items-center justify-center px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors w-full"
              >
                <FaRedo className="mr-2" />
                Take New Photo
              </button>
            )}
          </div>
          
          {detectionError && !capturedImage && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-md bg-yellow-100 border-2 border-yellow-300 text-yellow-800 text-sm font-medium"
            >
              <div className="flex items-start">
                <FaExclamationTriangle className="mt-0.5 mr-2 text-yellow-600" size={18} />
                <div>
                  <p className="font-bold mb-1">Face Detection Issue</p>
                  <p>{detectionError}</p>
                  <p className="mt-2 text-sm">Try using the "Force Register" button below if your face isn't being detected.</p>
                </div>
              </div>
            </motion.div>
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
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 shadow-sm">
          <h3 className="text-lg font-medium text-gray-800 mb-2">Instructions</h3>
          <ul className="space-y-2 text-gray-600">
            <motion.li 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="flex items-start"
            >
              <span className="text-primary-600 mr-2">1.</span>
              Enter the person's name in the input field
            </motion.li>
            <motion.li 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="flex items-start"
            >
              <span className="text-primary-600 mr-2">2.</span>
              Position your face in the center of the camera
            </motion.li>
            <motion.li 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="flex items-start"
            >
              <span className="text-primary-600 mr-2">3.</span>
              Ensure good lighting for better recognition
            </motion.li>
            <motion.li 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="flex items-start"
            >
              <span className="text-primary-600 mr-2">4.</span>
              Click "Capture & Register" to save your face
            </motion.li>
            <motion.li 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
              className="flex items-start"
            >
              <span className="text-primary-600 mr-2">5.</span>
              Wait for confirmation that registration is complete
            </motion.li>
          </ul>
          
          <div className="mt-6 p-3 bg-blue-50 border border-blue-100 rounded-md text-blue-800 text-sm">
            <p className="font-medium mb-1">Tips for better recognition:</p>
            <ul className="list-disc pl-5 space-y-1">
              <li>Face the camera directly</li>
              <li>Ensure even lighting on your face</li>
              <li>Avoid wearing accessories that cover your face</li>
              <li>Keep a neutral expression</li>
              <li>Use high quality setting for better accuracy</li>
            </ul>
          </div>
          
          {registrationStatus && (
            <div 
              className={`mt-4 p-4 rounded-lg text-sm ${
                registrationStatus.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}
            >
              <div className="flex items-start">
                {registrationStatus.success ? (
                  <FaCheck className="mr-2 mt-0.5 flex-shrink-0" />
                ) : (
                  <FaExclamationTriangle className="mr-2 mt-0.5 flex-shrink-0" />
                )}
                <div>
                  <p className="font-medium">
                    {registrationStatus.success ? 'Registration Successful' : 'Registration Failed'}
                  </p>
                  <p>{registrationStatus.message}</p>
                  {!registrationStatus.success && detectionError && (
                    <p className="mt-2 text-xs">
                      Error detail: {detectionError}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FaceRegistration; 