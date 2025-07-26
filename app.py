import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model import HeartRiskModel

# создаём директорию для временных загрузок
TMP_DIR = 'tmp'
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI()

app.mount("/tmp", StaticFiles(directory=TMP_DIR), name="tmp")

templates = Jinja2Templates(directory="templates")
model = HeartRiskModel(model_path='model.pkl', threshold=0.4)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/")
def form(request: Request):
    # простейшая страница с формой загрузки CSV
    return templates.TemplateResponse("start_form.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # сохраняем файл
    upload_path = os.path.join(TMP_DIR, file.filename)
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # делаем предсказание
    try:
        high_risk = model.predict(upload_path)
    except Exception as e:
        # при ошибке редиректим обратно с сообщением
        return RedirectResponse(url=f"/?error={e}", status_code=303)

    return templates.TemplateResponse(
        "res_form.html",
        {
            "request": request,
            "high_risk_ids": high_risk,
            "file_name": file.filename
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
