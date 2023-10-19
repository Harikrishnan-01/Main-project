import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import './App.css';
import Home from './components/Home.js';

const socket = io('http://127.0.0.1:8000');

const App = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [signText, setSignText] = useState('');
  const [processedFrame, setProcessedFrame] = useState('');

  const videoRef = useRef(null);
  const frameIntervalRef = useRef(null);

  const handleVideoClick = () => {
    // Play the video
    videoRef.current.play();
  };

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error('Error accessing camera:', error);
      }
    };

    startCamera();

    return () => {
      // Cleanup - stop the camera stream and frame interval when the component unmounts
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
      }
      clearInterval(frameIntervalRef.current);
    };
  }, []);

  const sendFrameToBackend = async (image, signLanguage) => {
    socket.emit('process_frame', { image, sign_language: signLanguage });

    socket.on('process_frame_result', (data) => {
      console.log(data.signText);

      setSignText(data.signText);
      setProcessedFrame(data.encodedFrame);
    });

    socket.on('process_frame_error', (data) => {
      console.error('Error processing frame:', data.error);
      setIsProcessing(false);
    });
  };

  const startFrameProcessing = () => {
    setIsProcessing(true);

    const processFrame = () => {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const image = canvas.toDataURL('image/jpeg');
      const signLanguage = document.getElementById('signLanguage').value;

      sendFrameToBackend(image, signLanguage);
    };

    processFrame();

    frameIntervalRef.current = setInterval(processFrame, 600);
  };

  const stopFrameProcessing = () => {
    clearInterval(frameIntervalRef.current);
    setProcessedFrame('');
    setIsProcessing(false);
  };

  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  const clearText = () => {
    const signLanguage = document.getElementById('signLanguage').value;
    fetch('http://127.0.0.1:8000/clear_text', {
      method: 'POST',
      body: JSON.stringify({ sign_language: signLanguage }),
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('Cleared text:', data.clearedText);
        setSignText('');
      })
      .catch((error) => {
        console.error('Error clearing text:', error);
      });
  };

  return (
    <div>
    <Home/>
    <div id='features-sec' className="container" >
      
      <h1>Sign Language Recognition</h1>

      <div className="predictionContainer">
        <label htmlFor="signLanguage">Select Sign Language:</label>
        <select id="signLanguage" className="signLabel">
          <option value="indian">Indian Sign Language</option>
          <option value="american">American Sign Language</option>
        </select>
      </div>

      <div className="buttonContainer">
        
          <div className="speakButton">
            <button onClick={() => speakText(signText)}>Speak</button>
          </div>
        {isProcessing ? (
          <div>
            <button onClick={stopFrameProcessing} disabled={!isProcessing}>
              Stop
            </button>
            <button onClick={clearText}>Clear Text</button>
          </div>
        ) : (
          <div>
            <button onClick={startFrameProcessing} disabled={isProcessing}>
              Start 
            </button>
            <button onClick={clearText}>Clear Text</button>
          </div>
        )}
      </div>
      
      <div className='resultSection'>
      

      
        <div className="predictionResultContainer">
          <h2>Sign Language Prediction</h2>
          
            <div>
            {signText ? (
                  <textarea className='result-text' style={{ width: '760px', height: '460px', fontSize:'2em' }} value={signText}></textarea>

                ) : (
                  <textarea className='result-text' style={{ width: '760px', height: '460px', fontSize:'2em' }} value={signText}></textarea>

                )}

          </div>
        
        </div>
      
        
        <div className='processedFullFrameContainer'>
          <h2>Processed Frame</h2>
          <div className="processedFrameContainer">
          
          {processedFrame && (
          <img src={`data:image/jpeg;base64,${processedFrame}`} alt="Processed Frame" />
          )}
          </div>
        </div>
      
        </div>
      <button onClick={handleVideoClick} style={{ display: 'none' }}>
        Play Video
      </button>

      <div className="showVideo">
        <video ref={videoRef} autoPlay playsInline muted></video>
      </div>
    </div>
    </div>
  );
};

export default App;
