# Live Facial Sentiment Analyser

## Intro

Hi there, thank you for taking the time to look over our project. This repository holds the frontend and backend for our web application, which allows you to try out our live facial sentiment analysis algorithm, or to upload an image and get a single sentiment analysis prediction.

Here is a demonstration of the project:

[![Alt text for your video](http://img.youtube.com/vi/9UWd0DgQwjM/0.jpg)](http://www.youtube.com/watch?v=9UWd0DgQwjM)

---

You can try out the web application yourselves at https://ml-final-project.vercel.app/

This project was created by Mekael Wasti and Gavin Bosman, for their CSCI-4050U final project.

## Running Locally

In order to run this project you will need Node.js and npm installed. First, in the root project folder run:

### 'npm install'

This will install all dependencies for the project. Next (still in the root folder) run:

### 'npm start'

and this should spin up the project in the browser. From there you can test out our model and play around with different images.
To view the pytorch backend, simply navigate to the /backend/ folder and you can run the jupyter notebook there.

You also need to run the backend API in a new terminal.
Open up the new terminal and cd into the Backend directory
then run

### 'uvicorn main:app --reload --host 0.0.0.0 --port 3001' or 'python -m uvicorn main:app --reload --host 0.0.0.0 --port 3001'

then in the App.js file, comment out this line (line 27)

### 'webSocket.current = new WebSocket("wss://api.mekaelwasti.com:63030/ws");'

and uncomment this line (line 28)

### 'webSocket.current = new WebSocket("ws://localhost:3001/ws ");'

Now it should run locally
