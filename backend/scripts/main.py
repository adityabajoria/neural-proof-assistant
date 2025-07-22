from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class GoalInput(BaseModel):
    goal: str

@app.get("/")
def home():
    return {"message": "Neural Proof Assistant backend is running!"}

@app.post("/predict")
def predict(goal_input: GoalInput):
    print("Received:", goal_input.goal)
    return {"tactic": "induction" if "∀" in goal_input.goal else "rewrite"}
