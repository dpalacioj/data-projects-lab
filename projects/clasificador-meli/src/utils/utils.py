import json

def load_model_config(path):
    with open(path) as f:
        return json.load(f)

def save_model_config(config, path):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def predicted_values(preds):
    pred_val = preds.tolist()[0]
    if pred_val == 0:
        return 'New'
    else:
        return 'Used'