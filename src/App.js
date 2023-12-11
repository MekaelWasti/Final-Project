import logo from "./logo.svg";
import "./App.css";
import react from "react";
import React, { useState, useRef, useEffect } from "react";

function App() {
  // States
  const [isCameraActive, setIsCameraActive] = useState(true);
  const [sentiment, setSentiment] = useState("Upload Image");
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  // VideoRefs
  const videoRef = useRef(null);
  const webSocket = useRef(null);

  const initializeWebSocket = () => {
    webSocket.current = new WebSocket("wss://api.mekaelwasti.com:63030/ws");
    webSocket.current.onopen = () => console.log("WebSocket Connected");
    webSocket.current.onmessage = (message) => {
      const sentimentData = JSON.parse(message.data);
      const percentage = parseInt(Math.round(sentimentData.sentiment[1]));
      let color = "";
      if (percentage > 70) {
        color = "#00FF09"; // Green
      } else if (percentage < 40) {
        color = "#FF0000"; // Red
      } else {
        color = "#FFFF00"; // Yellow
      }

      setSentiment({
        text: `${sentimentData.sentiment[0].toUpperCase()}: ${percentage}%`,
        color: color,
      });
    };
    webSocket.current.onclose = () => console.log("WebSocket Disconnected");
  };

  // Function to toggle camera
  const toggleCamera = () => {
    if (isCameraActive) {
      // Stop the webcam and close WebSocket
      if (videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      if (webSocket.current) {
        webSocket.current.close();
      }
    } else {
      // Start the webcam and initialize WebSocket
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            initializeWebSocket(); // Initialize WebSocket
          }
        })
        .catch((error) => {
          console.error("Error accessing camera.", error);
        });
    }
    setIsCameraActive(!isCameraActive);
  };

  // Function to capture and send frame
  const captureFrame = () => {
    if (!videoRef.current || !webSocket.current) return;

    if (webSocket.current.readyState === WebSocket.OPEN) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
      const data = canvas.toDataURL("image/jpeg");
      webSocket.current.send(data);
    } else {
      console.log("WebSocket is not ready for sending data.");
    }
  };

  // Rate of sending frames
  useEffect(() => {
    if (isCameraActive) {
      const interval = setInterval(captureFrame, 100);
      return () => clearInterval(interval);
    }
  }, [isCameraActive]);

  useEffect(() => {
    if (isCameraActive) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            initializeWebSocket(); // Initialize WebSocket
          }
        })
        .catch((error) => {
          console.error("Error accessing media devices.", error);
        });

      const interval = setInterval(captureFrame, 100);
      return () => clearInterval(interval);
    }
  }, [isCameraActive]);

  // Image upload handler
  const handleImageSubmission = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileField = document.querySelector("input[type=file]");
    formData.append("image", fileField.files[0]);

    try {
      const response = await fetch(
        "https://api.mekaelwasti.com:63030/upload_image",
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const sentimentData = await response.json();
      const percentage = parseInt(Math.round(sentimentData[1]));
      let color = "";
      if (percentage > 70) {
        color = "#00FF09"; // Green
      } else if (percentage < 40) {
        color = "#FF0000"; // Red
      } else {
        color = "#FFFF00"; // Yellow
      }

      setSentiment({
        text: `${sentimentData[0].toUpperCase()}: ${percentage}%`,
        color: color,
      });
    } catch (err) {
      console.log("Error:", err);
    }
  };

  return (
    <div className="App">
      <h1 id="MainHeader">
        EXPRESS <br></br>FACIAL SENTIMENT ANALYSER
      </h1>
      {isCameraActive? null :
      <h1>
          SENTIMENT:{" "}
          <span style={{ color: sentiment.color }}>{sentiment.text}</span>
      </h1>}

      <hr></hr>
      <h2>IMAGE ANALYSIS</h2>
      <h3>UPLOAD YOUR IMAGE HERE</h3>
      <div className="file_upload">
        <form onSubmit={handleImageSubmission} encType="multipart/form-data">
          <label htmlFor="image" className="label">
            <div id="labelRow"> </div>
            <input
              type="file"
              name="image"
              id="image"
              accept="image/jpeg,image/jpg,image/png,image/gif"
              onChange={handleImageChange} // Add onChange handler
            />
          </label>
          <br></br>
          <input className="button" type="submit" value="UPLOAD" />
        </form>
        {/* Image Preview */}
        {imagePreviewUrl && (
          <img
            src={imagePreviewUrl}
            alt="Image Preview"
            style={{ width: "50%", height: "auto" }}
          />
        )}
      </div>
      <hr></hr>
      <div className="video">
        <h2>LIVE VIDEO ANALYSIS</h2>
        <button className="button" onClick={toggleCamera}>
          {isCameraActive ? "TURN OFF CAMERA" : "TURN ON CAMERA"}
        </button>
        {isCameraActive? <h1>
          SENTIMENT:{" "}
          <span style={{ color: sentiment.color }}>{sentiment.text}</span>
        </h1> : null}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          style={{ width: "50%", height: "50%" }}
        />
      </div>
    </div>
  );
}

export default App;
