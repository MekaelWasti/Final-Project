import logo from "./logo.svg";
import "./App.css";
import react from "react";
import React, { useState, useRef, useEffect } from "react";

function App() {
  // States
  const [sentiment, setSentiment] = useState("Upload Image");

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
      <h1>EXPRESS - FACIAL SENTIMENT ANALYSER</h1>
      <div className="file_upload">
        <form onSubmit={handleImageSubmission} encType="multipart/form-data">
          <label htmlFor="image">
            {" "}
            <h4>UPLOAD YOUR IMAGE HERE</h4>
            <h4>SENTIMENT: {sentiment}</h4>
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
      {/* <video
        ref={videoRef}
        autoPlay
        style={{ width: "640px", height: "480px" }}
      /> */}
    </div>
  );
}

export default App;
