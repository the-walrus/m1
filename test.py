import requests

def main():
    # проверим, что сервис запущен
    r = requests.get("http://localhost:8000/health")
    assert r.status_code == 200
    print("Health OK:", r.json())

    # проверим предсказание на заглушечном файле submission.csv
    files = {'file': open('submission.csv', 'rb')}
    r2 = requests.post("http://localhost:8000/predict", files=files)
    print("Predict status:", r2.status_code)
    # в случае html-ответа можно печатать часть текста
    print("Response snippet:", r2.text[:200])

if __name__ == "__main__":
    main()
