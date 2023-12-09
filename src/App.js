import logo from "./logo.svg";
import "./App.css";
import react from "react";

function App() {
  const handleImageSubmission = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileField = document.querySelector("input[type=file]");
    formData.append("image", fileField.files[0]);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload_image", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const result = await response.json();
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
    </div>
  );
}

export default App;
