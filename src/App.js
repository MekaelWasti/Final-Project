import logo from "./logo.svg";
import "./App.css";
import react from "react";
import React, { useState, useRef, useEffect } from "react";

function App() {
  // States
  const [sentiment, setSentiment] = useState("Upload Image");

  // VideoRefs
  const videoRef = useRef(null);
  const webSocket = useRef(null);

  useEffect(() => {
    // Request access to the webcam
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          // Set the source of the video element to the stream from the webcam
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch((error) => {
          console.error("Error accessing media devices.", error);
        });
    } else {
      console.error(
        "MediaDevices API or getUserMedia method is not supported in this browser."
      );
    }
  }, []);

  useEffect(() => {
    // Establish WebSocket connection
    webSocket.current = new WebSocket("wss://api.mekaelwasti.com:63030/ws");
    webSocket.current.onopen = () => {
      console.log("WebSocket Connected");
    };
    webSocket.current.onmessage = (message) => {
      const data = JSON.parse(message.data);
      const sentiment = data.sentiment;
      console.log(sentiment);
      setSentiment(sentiment.toUpperCase());
    };
    webSocket.current.onclose = () => {
      console.log("WebSocket Disconnected");
    };
    // Clean
    return () => {
      if (webSocket.current) {
        webSocket.current.close();
      }
    };
  }, []);

  const captureFrame = () => {
    if (!videoRef.current) return;

    if (webSocket.current && webSocket.current.readyState === WebSocket.OPEN) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
      const data = canvas.toDataURL("image/jpeg");
      webSocket.current.send(data);
    } else {
      // Optionally handle the case where the WebSocket is not ready
      console.log("WebSocket is not ready for sending data.");
    }
  };

  // Rate of sending frames
  useEffect(() => {
    const interval = setInterval(captureFrame, 100);
    return () => clearInterval(interval);
  }, []);

  // Image upload handler
  const handleImageSubmission = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileField = document.querySelector("input[type=file]");
    formData.append("image", fileField.files[0]);

    try {
      // const response = await fetch("http://127.0.0.1:8000/upload_image", {
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

      const result = await response.json();
      setSentiment(result.toUpperCase());
      console.log("Image sent to backend", result);
    } catch (err) {
      console.log("Error:", err);
    }
  };
  return (
    <div className="App">
      <h1 id="MainHeader">EXPRESS - FACIAL SENTIMENT ANALYSER</h1>
      <div className="file_upload">
        <form onSubmit={handleImageSubmission} encType="multipart/form-data">
          <label htmlFor="image" className="label">
            <div id="labelRow">
              {" "}
              <h4>UPLOAD YOUR IMAGE HERE</h4>
              <h4>SENTIMENT: {sentiment}</h4>
            </div>
          </label>
          <br></br>
          <input
            type="file"
            name="image"
            id="image"
            accept="image/jpeg,image/jpg,image/png,image/gif"
          />
          <br></br>
          <input type="submit" value="UPLOAD" />
        </form>
      </div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "50%", height: "50%" }}
      />
    </div>
  );
}

export default App;
