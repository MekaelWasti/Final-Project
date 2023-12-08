import logo from "./logo.svg";
import "./App.css";

function App() {
  return (
    <div className="App">
      <h1>EXPRESS - FACIAL SENTIMENT ANALYSER</h1>
      <div className="file_upload">
        <form action="upload_image" method="POST" enctype="multipart/form-data">
          <label for="image">
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
