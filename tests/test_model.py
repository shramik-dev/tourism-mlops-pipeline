
def test_model_load():
    import joblib
    model = joblib.load('models/model.joblib')
    assert model is not None
    print("Test passed")
