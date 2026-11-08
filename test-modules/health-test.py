import json

# read the json file from the previous step
with open("health-response.json", encoding="utf-8") as file:
    response = json.load(file)

    print(response)
    assert response.get("name") == "Loan Amount Prediction API"