# python3.10 -m uvicorn main:app --reload --host 0.0.0.0 --port 63030 --ssl-keyfile=./ZERO_SSL/private.key --ssl-certfile=./ZERO_SSL/certificate.crt

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
app = FastAPI()
from inference import getSentiment


# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:3001",
    "https://thesisproject.mekaelwasti.com",
    "https://thesisproject.mekaelwasti.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.post("/upload_audio/")
# async def upload_audio(file: UploadFile = File(...)):
#     try:
#         with open("uploaded_file.wav", "wb") as buffer:
#             buffer.write(file.file.read())

#         transcription = VoiceRecognition("uploaded_file.wav")
#         # res = getCandidate(transcription)
#         res = getCandidateV2(transcription)
#         print("GET CANDIDATE: ", res)

#         # Convert res to a string and include it in the message of the response
#         return JSONResponse(content={"message": f"{str(res)}"}, status_code=200)
#         # return PlainTextResponse(content=res, status_code=200)

#     except Exception as e:
#         return JSONResponse(content={"message": f"Error: {e}"}, status_code=500)
#         # return PlainTextResponse(content=f"Error: {e}", status_code=500)


@app.post("/upload_image")
async def getAction(image: UploadFile = File(...)):
    print("AH YEAH")
    res = await image.read()
    getSentiment(res)

    

    # with open(f"{file.filename}", "wb") as buffer:
        # shutil.copyfileobj(file.file, buffer)
    # return {"filename": file.filename}
    # try:
        # return JSONResponse(content={"message":"Image uploaded"})
    # except Exception as e:
        # return JSONResponse(content={"Error":str(e)}, status_code=500)



#     print(f'RECIEVED: {userInput}')
#     # res = getSentiment(userInput)
#     # print("RES:", res)
#     # return res