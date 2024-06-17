from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates/")

# Render the form on the index page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Handle form submission and redirect to welcome page
@app.post("/welcome", response_class=HTMLResponse)
async def push(request: Request, fname: str = Form(...),lname: str = Form(...)):
    return RedirectResponse(url=f'/welcomepage?fname={fname}&lname={lname}', status_code=303)

# Render the welcome page with the name
@app.get("/welcomepage", response_class=HTMLResponse)
async def welcomed(request: Request,fname, lname):
    return templates.TemplateResponse("welcomeDone.html", {"request": request, "fname": fname, "lname":lname})
