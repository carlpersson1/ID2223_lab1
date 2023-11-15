import gradio as gr
import hopsworks
import pandas as pd
import keras

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()

n_input = 7 # Number of features
n_hidden = 256  # Number of hidden nodes
n_out = 3 # Number of classes
model = keras.Sequential([
    keras.layers.Dense(n_hidden, input_dim=n_input, activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(n_hidden, activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(n_hidden, activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(n_hidden, activation='selu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(n_hidden, activation='selu'),
    keras.layers.Dense(n_out, activation='softmax')
])

model.load_weights(model_dir + '/wine_model.h5')
print("Model downloaded")


def wine(type, alcohol, density, citric_acid, vol_acid, chlorides, total_sulfur):
    print("Calling function")
    type = 0 if type == 'White' else 1

#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[type, vol_acid, citric_acid, chlorides, total_sulfur, density, alcohol]],
                      columns=['type', 'volatile acidity', 'citric acid', 'chlorides', 'total sulfur dioxide', '´density', 'alcohol'])
    print("Predicting")
    # 'res' is a list of predictions returned as the label.
    wine_prediction = model.predict(df)
    wine_prediction = {'High Quality': float(wine_prediction[0][2]),
                       'Good Quality': float(wine_prediction[0][1]),
                       'Low Quality': float(wine_prediction[0][0])}
    return wine_prediction


demo = gr.Interface(
    fn=wine,
    title="Wine Quality Analytics",
    description="Experiment with wine contents to predict what quality of wine it is.",
    allow_flagging="never",
    inputs=[
        gr.Radio(["White", "Red"], value='White', label="What kind of wine is it?"),
        gr.Number(value=12.8, label="Alcohol content (%), normal range 8-15 %"),
        gr.Number(value=0.9892, label="Density (kg/dm^3), normal range 0.99-0.101 kg/dm^3"),
        gr.Number(value=0.48, label="Citric acid content (g/L), normal range 0.0-1.7 g/L"),
        gr.Number(value=0.66, label="Volatile acid content (g/L), normal range 0.1 - 1.6 g/L"),
        gr.Number(value=0.029, label="Chloride content (g/L), normal range 0.01-0.6 g/L"),
        gr.Number(value=75., label="Total sulfur dioxide content (ppm), normal range 6-450 ppm"),
        ],
    outputs=gr.Label(num_top_classes=3))

demo.launch(debug=True)

